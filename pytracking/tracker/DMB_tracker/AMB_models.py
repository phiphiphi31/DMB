import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import cv2
from torchvision import models
import seaborn as sns
import matplotlib.pyplot as plt

def mask_iou(pred, target):

    """
    param: pred of size [N x H x W]
    param: target of size [N x H x W]
    """

    assert len(pred.shape) == 3 and pred.shape == target.shape

    N = pred.size(0)

    inter = torch.min(pred, target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    iou = torch.sum(inter / union) / N

    return iou

def Soft_aggregation(ps, max_obj):
    
    num_objects, H, W = ps.shape
    em = torch.zeros(1, max_obj+1, H, W).to(ps.device)
    em[0, 0, :, :] =  torch.prod(1-ps, dim=0) # bg prob
    em[0,1:num_objects+1, :, :] = ps # obj prob
    em = torch.clamp(em, 1e-7, 1-1e-7)
    logit = torch.log((em /(1-em)))

    return logit

class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride==1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
 
        if self.downsample is not None:
            x = self.downsample(x)
         
        return x + r 

class Encoder_M(nn.Module):
    def __init__(self):
        super(Encoder_M, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_o = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/16, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f, in_m, in_bg):
        # f = (in_f - self.mean) / self.std
        f = in_f # [2, 3, 384, 384]
        m = torch.unsqueeze(in_m, dim=1).float() # add channel dim [2, 1, 384, 384]
        bg = torch.unsqueeze(in_bg, dim=1).float() # add channel dim [2, 1, 384, 384]

        x = self.conv1(f) + self.conv1_m(m) + self.conv1_o(bg) # [2, 64, 192, 192]
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/16, 1024 [2, 1024. 24, 24]

        return r4, r3, r2, c1
 
class Encoder_Q(nn.Module):
    def __init__(self):
        super(Encoder_Q, self).__init__()

        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/16, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f, output_layers=None):
        # f = (in_f - self.mean) / self.std
        f = in_f

        x = self.conv1(f) 
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/16, 1024 # [1, 1024, 24, 24]

        return r4, r3, r2, c1


class Refine(nn.Module):
    def __init__(self, inplanes, planes):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, size=s.shape[2:], mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m

class Decoder(nn.Module):
    def __init__(self, inplane, mdim):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(inplane, mdim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim) # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim) # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, r4, r3, r2, f, test_dist = None):


        m4 = self.ResMM(self.convFM(r4)) # [1, 256, 24, 24]

        if test_dist is not None:
            dist = test_dist / test_dist.shape[-1] # norm
            dist = F.interpolate(dist.unsqueeze(0), size=(m4.shape[-2], m4.shape[-1])) # [1, 1, 24, 24]
        # position encoding

        #r3_pe = r3 +  dist
        #m3 = self.RF3(r3_pe, m4) # out: 1/8, 256 [1, 256, 48, 48]

        m4_pe = m4 + dist
        m3 = self.RF3(r3, m4_pe) # out: 1/8, 256 [1, 256, 48, 48]

        m2 = self.RF2(r2, m3) # out: 1/4, 256 [1, 256, 96, 96]

        p2 = self.pred2(F.relu(m2)) # [1, 2, 96, 96]

        p = F.interpolate(p2, size=f.shape[2:], mode='bilinear', align_corners=False) # # [1, 2, 384, 384]
        return p

class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()
 
    def forward(self, m_in, m_out, q_in, q_out):  # m_in: o,c,t,h,w
        _, _, H, W = q_in.size()
        no, centers, C = m_in.size() #[num_obj, THW, dim]
        _, _, vd = m_out.shape
 
        qi = q_in.view(-1, C, H*W) #[num_obj, 128, hw]

        p = torch.bmm(m_in, qi) # no x centers x hw
        p = p / math.sqrt(C)
        p = torch.softmax(p, dim=1) # no x centers x hw

        mo = m_out.permute(0, 2, 1) # no x c x centers 
        mem = torch.bmm(mo, p) # no x c x hw
        mem = mem.view(no, vd, H, W) # [num_obj, 1024, h, w]

        mem_out = torch.cat([mem, q_out], dim=1)

        return mem_out, p

class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        # self.Key = nn.Linear(indim, keydim)
        # self.Value = nn.Linear(indim, valdim)
        self.Key = nn.Conv2d(indim, keydim, kernel_size=3, padding=1, stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=3, padding=1, stride=1)
 
    def forward(self, x):  
        return self.Key(x), self.Value(x)

class AMB(nn.Module):
    def __init__(self, keydim = 128, valdim = 512, phase='test', mode='recurrent', iou_threshold=0.5):
        super(AMB, self).__init__()
        self.Encoder_M = Encoder_M() 
        self.Encoder_Q = Encoder_Q()

        self.keydim = keydim
        self.valdim = valdim

        self.KV_M_r4 = KeyValue(1024, keydim=keydim, valdim=valdim)
        self.KV_Q_r4 = KeyValue(1024, keydim=keydim, valdim=valdim)
        # self.Routine = DynamicRoutine(channel, iters, centers)

        self.Memory = Memory()
        self.Decoder = Decoder(2*valdim, 256)
        self.phase = phase
        self.mode = mode
        self.iou_threshold = iou_threshold

        assert self.phase in ['train', 'test']

    def load_param(self, weight):

        s = self.state_dict()
        if 'state_dict' in weight.keys():
            weight = weight['state_dict']

        for key, val in weight.items():

            # process ckpt from parallel module
            if key[:6] == 'module':
                key = key[7:]

            if key in s and s[key].shape == val.shape:
                s[key][...] = val
            elif key not in s:
                print('ignore weight from not found key {}'.format(key))
            else:
                print('ignore weight of mistached shape in key {}'.format(key))

        self.load_state_dict(s)

    def memorize(self, frame, masks, num_objects = 1):
        # memorize a frame 
        # maskb = prob[:, :num_objects, :, :]
        # make batch arg list
        frame_batch = []
        mask_batch = []
        bg_batch = []
        #bg_batch_orig = []

        # print('\n')
        # print(num_objects)

        # frame [1, 3, set_w, set_h]
        # masks [1, 6, set_w, set_h]
        try:
            for o in range(1, num_objects+1): # 1 - no
                frame_batch.append(frame)
                mask_batch.append(masks[:,o])

            for o in range(1, num_objects+1):
                bg_batch.append(torch.clamp(1.0 - masks[:, o], min=0.0, max=1.0))
               # bg_batch_orig.append((torch.sum(masks[:, 1:o], dim=1) + \
                             #       torch.sum(masks[:, o + 1:num_objects + 1], dim=1)).clamp(0, 1))  # [1, 0, h ,w ]

            # make Batch
            frame_batch = torch.cat(frame_batch, dim=0) # [num_objects, 3, set_w, set_h]
            mask_batch = torch.cat(mask_batch, dim=0) # [num_objects, set_w, set_h]
            bg_batch = torch.cat(bg_batch, dim=0)
            #bg_batch_orig = torch.cat(bg_batch_orig, dim=0)

        except RuntimeError as re:
            print(re)
            print(num_objects)
            raise re

        r4, _, _, _ = self.Encoder_M(frame_batch, mask_batch, bg_batch) # no, c, h, w  [2, 1024, 24, 24]
        _, c, h, w = r4.size()
        memfeat = r4
        # memfeat = self.Routine(memfeat, maskb)
        # memfeat = memfeat.view(-1, c)
        k4, v4 = self.KV_M_r4(memfeat) # num_objects, 128 and 512, H/16, W/16
        k4 = k4.permute(0, 2, 3, 1).contiguous().view(num_objects, -1, self.keydim) # [2, h*w, 128]
        v4 = v4.permute(0, 2, 3, 1).contiguous().view(num_objects, -1, self.valdim)
        
        return k4, v4, r4

    def segment(self, frame, keys, values,  test_dist = None):
        # segment one input frame
        num_objects = 1
        max_obj = 1

        r4, r3, r2, _ = self.Encoder_Q(frame) # [1, 1024, 24, 24]
        n, c, h, w = r4.size()
        # r4 = r4.permute(0, 2, 3, 1).contiguous().view(-1, c)
        k4, v4 = self.KV_Q_r4(r4)   # 1, dim, H/16, W/16
        # k4 = k4.view(n, self.keydim, -1).permute(0, 2, 1)
        # v4 = v4.view(n, self.valdim, -1).permute(0, 2, 1)

        # expand to ---  no, c, h, w
        k4e, v4e = k4.expand(num_objects,-1,-1,-1), v4.expand(num_objects,-1,-1,-1) 
        r3e, r2e = r3.expand(num_objects,-1,-1,-1), r2.expand(num_objects,-1,-1,-1)

        m4, _ = self.Memory(keys, values, k4e, v4e) #[num, 1024, 24, 24] #[num, 1024, 24, 24]
        logit = self.Decoder(m4, r3e, r2e, frame, test_dist = test_dist ) # [num_objects, 2, 480, 912] test_dist[1, 384, 384]
        ps = F.softmax(logit, dim=1)[:, 1] # no, h, w  
        # ps = torch.sigmoid(logit)[:, 1]
        #ps = indipendant possibility to belong to each object


        logit = Soft_aggregation(ps, max_obj) # 1, K, H, W

        return logit, ps
    # test_dist: reference [1, 384, 384] train: [B, 3, 384, 384]
    def forward(self, frame, mask=None, keys=None, values=None, num_objects=None, max_obj=None, test_dist = None):

        if self.phase == 'test':
            if mask is not None: # keys
                return self.memorize(frame, mask, num_objects)
            else:
                return self.segment(frame, keys, values, num_objects, max_obj, test_dist)
        elif self.phase == 'train':

            N, T, C, H, W = frame.size()
            max_obj = mask.shape[2]-1

            total_loss = 0.0
            batch_out = []
            for idx in range(N):

                num_object = num_objects[idx].item()

                batch_keys = []
                batch_vals = []
                tmp_out = []
                for t in range(1, T):
                    # memorize
                    if t-1 == 0 or self.mode == 'mask':
                        tmp_mask = mask[idx, t-1:t]
                    elif self.mode == 'recurrent':
                        tmp_mask = out
                    else:
                        pred_mask = out[0, 1:num_object+1]
                        iou = mask_iou(pred_mask, mask[idx, t-1, 1:num_object+1])

                        if iou > self.iou_threshold:
                            tmp_mask = out
                        else:
                            tmp_mask = mask[idx, t-1:t]

                    key, val, _ = self.memorize(frame=frame[idx, t-1:t], masks=tmp_mask, 
                        num_objects=num_object)

                    batch_keys.append(key)
                    batch_vals.append(val)
                    # segment
                    tmp_key = torch.cat(batch_keys, dim=1)
                    tmp_val = torch.cat(batch_vals, dim=1)
                    # input test_dist = [1, 384, 384]
                    logits, ps = self.segment(frame=frame[idx, t:t+1], keys=tmp_key, values=tmp_val,
                        num_objects=num_object, max_obj=max_obj, test_dist = test_dist[idx, t:t+1]) # test_dist[b, 3, w, h]

                    out = torch.softmax(logits, dim=1)
                    tmp_out.append(out)
                
                batch_out.append(torch.cat(tmp_out, dim=0))

            batch_out = torch.stack(batch_out, dim=0) # B, T, No, H, W

            return batch_out

        else:
            raise NotImplementedError('unsupported forward mode %s' % self.phase)

    def pad_divide_by(self, in_list, d, in_size):
        out_list = []
        h, w = in_size
        if h % d > 0:
            new_h = h + d - h % d
        else:
            new_h = h
        if w % d > 0:
            new_w = w + d - w % d
        else:
            new_w = w
        lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
        lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
        pad_array = (int(lw), int(uw), int(lh), int(uh))
        for inp in in_list:
            out_list.append(F.pad(inp, pad_array))
        return out_list, pad_array