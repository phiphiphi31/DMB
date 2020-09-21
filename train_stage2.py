from libs.dataset.data import ROOT, DATA_CONTAINER, multibatch_collate_fn
from libs.dataset.transform import TrainTransform, TestTransform, TrainTransform_Noresize
from libs.utils.logger import Logger, AverageMeter
from libs.utils.loss import *
from libs.utils.utility import write_mask, save_checkpoint, adjust_learning_rate
from libs.models.models import AMB

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import numpy as np
import os
import os.path as osp
import shutil
import time
import pickle
import argparse
import random
from progress.bar import Bar
from collections import OrderedDict
import cv2
from options_ytvos_nobackbone import OPTION as opt

from libs.train_data import segm_processing, segm_sampler
from libs.train_data.loader import LTRLoader
from libs.train_data.vos import Vos
import torchvision.transforms
import libs.train_data.transforms as dltransforms



MAX_FLT = 1e6

def parse_args():
    parser = argparse.ArgumentParser('Training Mask Segmentation')
    parser.add_argument('--gpu', default='0, 1', type=str, help='set gpu id to train the network, split with comma')
    return parser.parse_args()

def main():

    settings_print_interval = 1  # How often to print loss and other info
    settings_batch_size = 4 # Batch size 80   default 64
    settings_num_workers = 16 # Number of workers for image loading
    settings_normalize_mean = [0.485, 0.456, 0.406]  # Normalize mean (default pytorch ImageNet values)
    settings_normalize_std = [0.229, 0.224, 0.225]  # Normalize std (default pytorch ImageNet values)
    settings_search_area_factor = 4.0  # Image patch size relative to target size
    settings_feature_sz = 24  # Size of feature map
    settings_output_sz = settings_feature_sz * 16  # Size of input image patches 24*16
    settings_segm_use_distance = True

    # Settings for the image sample and proposal generation
    settings_center_jitter_factor = {'train': 0, 'test1': 1.5, 'test2': 1.5}
    settings_scale_jitter_factor = {'train': 0, 'test1': 0.25, 'test2': 0.25}
####################################################################################################
    start_epoch = 0
    random.seed(0)

    args = parse_args()
    # Use GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.gpu != '' else str(opt.gpu_id)
    use_gpu = torch.cuda.is_available() and (args.gpu != '' or int(opt.gpu_id)) >= 0
    gpu_ids = [int(val) for val in args.gpu.split(',')]

    if not os.path.isdir(opt.checkpoint):
        os.makedirs(opt.checkpoint)

    # Data
    print('==> Preparing dataset')

    input_size = opt.input_size

    train_transformer = TrainTransform(size=input_size)
    #train_transformer = TrainTransform_Noresize()
    test_transformer = TestTransform(size=input_size)

    try:
        if isinstance(opt.trainset, list):
            datalist = []
            for dataset, freq, max_skip in zip(opt.trainset, opt.datafreq, opt.max_skip):
                ds = DATA_CONTAINER[dataset](
                    train=True, 
                    sampled_frames=opt.sampled_frames, 
                    transform=train_transformer, 
                    max_skip=max_skip, 
                    samples_per_video=opt.samples_per_video
                )
                datalist += [ds] * freq

            trainset = data.ConcatDataset(datalist)

        else:
            max_skip = opt.max_skip[0] if isinstance(opt.max_skip, list) else opt.max_skip
            trainset = DATA_CONTAINER[opt.trainset](
                train=True, 
                sampled_frames=opt.sampled_frames, 
                transform=train_transformer, 
                max_skip=max_skip, 
                samples_per_video=opt.samples_per_video
                )
    except KeyError as ke:
        print('[ERROR] invalide dataset name is encountered. The current acceptable datasets are:')
        print(list(DATA_CONTAINER.keys()))
        exit()

    testset = DATA_CONTAINER[opt.valset](
        train=False,
        transform=test_transformer,
        samples_per_video=1
        )

    trainloader = data.DataLoader(trainset, batch_size=opt.train_batch, shuffle=True, num_workers=opt.workers,
                                  collate_fn=multibatch_collate_fn, drop_last=True)

    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=opt.workers,
                                 collate_fn=multibatch_collate_fn)


    #########################################################################################
    vos_train = Vos(split='train')
    transform_train = torchvision.transforms.Compose([dltransforms.ToTensorAndJitter(0.2),
                                                      torchvision.transforms.Normalize(mean=settings_normalize_mean,
                                                                                       std=settings_normalize_std)])
    data_processing_train = segm_processing.SegmProcessing(search_area_factor=settings_search_area_factor,
                                                           output_sz=settings_output_sz,
                                                           center_jitter_factor=settings_center_jitter_factor,
                                                           scale_jitter_factor=settings_scale_jitter_factor,
                                                           mode='pair',
                                                           transform=transform_train,
                                                           use_distance=settings_segm_use_distance)
    dataset_train = segm_sampler.SegmSampler([vos_train], [1],
                                             samples_per_epoch=1000 * settings_batch_size * 8, max_gap=50,
                                             processing=data_processing_train)
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings_batch_size,
                             num_workers=settings_num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)

    #########################################################################################

    # Model
    print("==> creating model")

    net = AMB(opt.keydim, opt.valdim, 'train', mode=opt.mode, iou_threshold=opt.iou_threshold)
    print('    Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))


    net.eval()

    if use_gpu:
        net = net.cuda()

    assert opt.train_batch % len(gpu_ids) == 0
    net = nn.DataParallel(net, device_ids=gpu_ids, dim=0)

    # set training parameters
    #for p in net.parameters():
      #  p.requires_grad = True
    for name, param in net.named_parameters():
        #print(name)
        if 'Encoder' in name:
            param.requires_grad = False  # 冻结 backbone 梯度
        else:
            param.requires_grad = True

    criterion = None
    celoss = cross_entropy_loss

    if opt.loss == 'ce':
        criterion = celoss
    elif opt.loss == 'iou':
        criterion = mask_iou_loss
    elif opt.loss == 'both':
        criterion = lambda pred, target, obj: celoss(pred, target, obj) + mask_iou_loss(pred, target, obj)
    else:
        raise TypeError('unknown training loss %s' % opt.loss)

    optimizer = None
    
    if opt.solver == 'sgd':

        optimizer = optim.SGD(net.parameters(), lr=opt.learning_rate,
                        momentum=opt.momentum[0], weight_decay=opt.weight_decay)
    elif opt.solver == 'adam':

        optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate,
                        betas=opt.momentum, weight_decay=opt.weight_decay)
    else:
        raise TypeError('unkown solver type %s' % opt.solver)

    # Resume
    title = 'Appearance Memory Bank'
    minloss = float('inf')

    opt.checkpoint = osp.join(osp.join(opt.checkpoint, opt.valset))
    if not osp.exists(opt.checkpoint):
        os.mkdir(opt.checkpoint)

    if opt.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint {}'.format(opt.resume))
        assert os.path.isfile(opt.resume), 'Error: no checkpoint directory found!'
        # opt.checkpoint = os.path.dirname(opt.resume)
        checkpoint = torch.load(opt.resume)
        minloss = checkpoint['minloss']
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        skips = checkpoint['max_skip']
        
        try:
            if isinstance(skips, list):
                for idx, skip in enumerate(skips):
                    trainloader.dataset.datasets[idx].set_max_skip(skip)
            else:
                trainloader.dataset.set_max_skip(skip)
        except:
            print('[Warning] Initializing max skip fail')

        logger = Logger(os.path.join(opt.checkpoint, opt.mode+'_log.txt'), resume=True)
    else:
        if opt.initial:
            print('==> Initialize model with weight file {}'.format(opt.initial))
            weight = torch.load(opt.initial)
            if isinstance(weight, OrderedDict):
                net.module.load_param(weight)
            else:
                net.module.load_param(weight['state_dict'])

        logger = Logger(os.path.join(opt.checkpoint, opt.mode+'_log.txt'), resume=False)
        start_epoch = 0

    logger.set_items(['Epoch', 'LR', 'Train Loss'])

    # Train and val
    for epoch in range(start_epoch):
        adjust_learning_rate(optimizer, epoch, opt)

    for epoch in range(start_epoch, opt.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, opt.epochs, opt.learning_rate))
        adjust_learning_rate(optimizer, epoch, opt)

        net.module.phase = 'train'
        train_loss = train(loader_train, # loader_train trainloader
                           model=net,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           use_cuda=use_gpu,
                           iter_size=opt.iter_size,
                           mode=opt.mode,
                           threshold=opt.iou_threshold)

        if (epoch + 1) % opt.epoch_per_test == 0:
            net.module.phase = 'test'
            test_loss = test(testloader,
                            model=net.module,
                            criterion=criterion,
                            epoch=epoch,
                            use_cuda=use_gpu)

        # append logger file
        logger.log(epoch+1, opt.learning_rate, train_loss)

        # adjust max skip
        if (epoch + 1) % opt.epochs_per_increment == 0:
            if isinstance(trainloader.dataset, data.ConcatDataset):
                for dataset in trainloader.dataset.datasets:
                    dataset.increase_max_skip()
            else:
                trainloader.dataset.increase_max_skip()

        # save model
        is_best = train_loss <= minloss
        minloss = min(minloss, train_loss)
        skips = [ds.max_skip for ds in trainloader.dataset.datasets] \
                if isinstance(trainloader.dataset, data.ConcatDataset) \
                 else trainloader.dataset.max_skip

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'loss': train_loss,
            'minloss': minloss,
            'optimizer': optimizer.state_dict(),
            'max_skip': skips,
        }, epoch + 1, is_best, checkpoint=opt.checkpoint, filename=opt.mode)

    logger.close()

    print('minimum loss:')
    print(minloss)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda, iter_size, mode, threshold):
    # switch to train mode

    data_time = AverageMeter()
    loss = AverageMeter()
    maskiou = AverageMeter()

    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    optimizer.zero_grad()

    for batch_idx, data in enumerate(trainloader):
        # masks[b, 3, K, w, h]
        frames = torch.cat([data['train_images'].permute(1, 0, 2, 3).unsqueeze(1),data['test1_images'].permute(1, 0, 2, 3).unsqueeze(1),data['test2_images'].permute(1, 0, 2, 3).unsqueeze(1)], 1).contiguous()

        for s in ['train', 'test1', 'test2']:
            data[s + '_masks'] = data[s + '_masks'].permute(1, 0, 2, 3).unsqueeze(2)
            data[s + '_masks'] = torch.cat([1- data[s + '_masks'], data[s + '_masks']], 2).contiguous()

        masks = torch.cat([data['train_masks'],data['test1_masks'],data['test2_masks']], 1).contiguous()
        dist_map = torch.cat([data['train_dist'].permute(1, 0, 2, 3),data['test1_dist'].permute(1, 0, 2, 3),data['test2_dist'].permute(1, 0, 2, 3), ], 1).contiguous()
        # dist_map = [b, 3, 384, 384]
        objs = torch.ones(frames.shape[0]).cuda().int()
        frames = frames.cuda()
        masks = masks.cuda()
        objs = objs.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        objs[objs==0] = 1

        N, T, C, H, W = frames.size() # N = batch size
        max_obj = masks.shape[2]-1

        total_loss = 0.0 # frame [b=4, 3, 3, set_h, set_w]
        iou = 0.0
        out = model(frame=frames, mask=masks, num_objects=objs, test_dist = dist_map) #[b, 2, K, w, h]
        for idx in range(N):
            for t in range(1, T):
                gt = masks[idx, t:t+1]
                pred = out[idx, t-1: t]
                No = objs[idx].item()

                total_loss = total_loss + criterion(pred, gt, No)

                a = pred[:, 1, :, :].clone()
                b = gt[:, 1, :, :].clone()
                iou_this = mask_iou(a, b).item()  # tensor[1, 384, 384]
                iou = iou + iou_this

        total_loss = total_loss / (N * (T-1))
        mean_maskiou = iou / (N * (T-1))

        # record loss
        if total_loss.item() > 0.0:
            loss.update(total_loss.item(), 1)

        if mean_maskiou > 0.0:
            maskiou.update(mean_maskiou, 1)

        # compute gradient and do SGD step (divided by accumulated steps)
        total_loss /= iter_size
        total_loss.backward()

        if (batch_idx+1) % iter_size == 0:
            optimizer.step()
            model.zero_grad()

        # measure elapsed time
        end = time.time()
        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s |Loss: {loss:.5f}| maskiou:{maskiou:.5f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.val,
            loss=loss.avg,
            maskiou = maskiou.avg
        )
        bar.next()
    bar.finish()

    return loss.avg

def test(testloader, model, criterion, epoch, use_cuda):

    data_time = AverageMeter()

    bar = Bar('Processing', max=len(testloader))

    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):

            frames, masks, objs, infos = data

            if use_cuda:
                frames = frames.cuda()
                masks = masks.cuda()
                
            frames = frames[0]
            masks = masks[0]
            num_objects = objs[0]
            info = infos[0]
            max_obj = masks.shape[1]-1
            # compute output
            t1 = time.time()

            T, _, H, W = frames.shape
            pred = [masks[0:1]]
            keys = []
            vals = []
            for t in range(1, T):
                if t-1 == 0:
                    tmp_mask = masks[0:1]
                elif 'frame' in info and t-1 in info['frame']:
                    # start frame
                    mask_id = info['frame'].index(t-1)
                    tmp_mask = masks[mask_id:mask_id+1]
                    num_objects = max(num_objects, tmp_mask.max())
                else:
                    tmp_mask = out

                # memorize
                key, val, _ = model(frame=frames[t-1:t, :, :, :], mask=tmp_mask, num_objects=num_objects)

                # segment
                tmp_key = torch.cat(keys+[key], dim=1)
                tmp_val = torch.cat(vals+[val], dim=1)
 ########################################################################################################################
                # 产生 test distance map
                mask_np = masks[t, 1, :, : ].cpu().numpy().astype(np.uint8)
                boxes = cv2.boundingRect(mask_np)
                # use target center only to create distance map
                cx_ = (boxes[0] + boxes[2] / 2) + (
                        (0.25 * boxes[2]) * (random.random() - 0.5))
                cy_ = (boxes[1] + boxes[3] / 2) + (
                        (0.25 * boxes[3]) * (random.random() - 0.5))
                x_ = np.linspace(1, frames[t:t+1, :, :, :].shape[3], frames[t:t+1, :, :, :].shape[3]) - 1 - cx_
                y_ = np.linspace(1, frames[t:t+1, :, :, :].shape[2], frames[t:t+1, :, :, :].shape[2]) - 1 - cy_
                X, Y = np.meshgrid(x_, y_)
                D = np.sqrt(np.square(X) + np.square(Y)).astype(np.float32)

                # show2 = D
                ## sns.heatmap(show2)
                # plt.show()
                test_dist_this = torch.from_numpy(np.expand_dims(D, axis=0))
                test_dist_this = test_dist_this.cuda()
###########################################################################################################################
                logits, ps = model(frame=frames[t:t+1, :, :, :], keys=tmp_key, values=tmp_val, num_objects=num_objects, max_obj=max_obj, test_dist = test_dist_this )

                out = torch.softmax(logits, dim=1)
                pred.append(out)

                if (t-1) % opt.save_freq == 0:
                    keys.append(key)
                    vals.append(val)
            
            pred = torch.cat(pred, dim=0)
            pred = pred.detach().cpu().numpy()
            write_mask(pred, info, opt, directory=opt.output_dir)

            toc = time.time() - t1

            data_time.update(toc, 1)
           
            # plot progress
            bar.suffix  = '({batch}/{size}) Time: {data:.3f}s'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                data=data_time.sum
            )
            bar.next()
        bar.finish()

    return

if __name__ == '__main__':
    main()
