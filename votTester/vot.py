# -*- coding: utf-8 -*
import copy
import itertools
import logging
import math
import os
import os.path as osp
from collections import OrderedDict
from multiprocessing import Process, Queue
from os.path import join

import cv2
import numpy as np
from tqdm import tqdm
from votTester import vot_benchmark
from votTester.pysot.bin.eval import vot_eval

def read_image(image_file: str):
    return cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)

def ensure_dir(dir_path: str):
    r"""
    Ensure the existence of path (i.e. mkdir -p)
    Arguments
    ---------
    dir_path: str
        path to be ensured
    """
    if osp.exists(dir_path):
        return
    else:
        os.makedirs(dir_path)

"""
class vot_eval: __init__(self, dataset, dataset_dir, tracker_result_dir, num = 8 ):

"""


class VOTTester:
    r"""
    Tester to test the vot dataset, the result is saved as follows
    exp_dir/logs/$dataset_name$/$tracker_name/
                                    |-baseline/$video_name$/ folder of result files
                                    |-eval_result.csv evaluation result file

    Hyper-parameters
    ----------------
    device_num: int
        number of gpu for test
    data_root: dict
        vot dataset root directory. dict(dataset_name: path_to_root)
    dataset_names: str
        daataset name (VOT2018|VOT2019)
    """

    def __init__(self, tracker, test_name, dataset_names = ['vot2018'], *args, **kwargs):

        self.speed = -1
        self.result_root_dir = './votTester/results'
        self.data_root = {
            'vot2018':'/home/jaffe/Titanspace/datasets/dataset/testdataset/dataset/VOT/vot2018/VOT2018' ,
            'vot2019':'/home/jaffe/Titanspace/datasets/dataset/testdataset/dataset/VOT/vot2019/VOT2019',
            'vot2016':'/home/jaffe/Titanspace/datasets/dataset/testdataset/dataset/VOT/vot2016/VOT2016'

        }
        self.test_tracker = tracker
        self.dataset_names = dataset_names
        self.test_name = test_name
        self.pysot_eval = vot_eval(self.dataset_names[0], self.data_root[self.dataset_names[0]], os.path.join(self.result_root_dir,self.dataset_names[0]), num=8)


    def test(self, ):

        for dataset_name in self.dataset_names:
            self.dataset_name = dataset_name
            # self.tracker_dir = os.path.join(self._cfg.auto.log_dir, self._hyper_params["dataset_name"])
            self.save_root_dir = os.path.join(self.result_root_dir,self.dataset_name, self.test_name, "baseline")
            ensure_dir(self.save_root_dir)
            # track videos
            self.run_tracker()
            # evaluation
            save_root_dir = os.path.join(self.result_root_dir, self.dataset_name)
            vot_root = self.data_root[dataset_name]
            vot_eval_show = vot_eval(dataset_name, vot_root, save_root_dir, num=8)
            vot_eval_show.eval(self.test_name)

    def run_tracker(self):
        """
        Run self.pipeline on VOT
        """
        vot_root = self.data_root[self.dataset_name]
        # setup dataset
        dataset = vot_benchmark.load_dataset(vot_root, self.dataset_name)
        self.dataset = dataset
        keys = list(dataset.keys())
        keys.sort()
      #  keys = ['butterfly']
        nr_records = len(keys)
        pbar = tqdm(total=nr_records)
        mean_speed = -1
        total_lost = 0
        speed_list = []
        result_queue = Queue(500)
        speed_queue = Queue(500)
        # set worker
        self.worker(keys, result_queue, speed_queue)
        for i in range(nr_records):
            t = result_queue.get()
            s = speed_queue.get()
            total_lost += t
            speed_list.append(s)
            pbar.update(1)

        # print result
        mean_speed = float(np.mean(speed_list))
        print('Total Lost: {:d}'.format(total_lost))
        print('Mean Speed: {:.2f} FPS'.format(mean_speed))
        self.speed= mean_speed

    def worker(self, records, result_queue=None, speed_queue=None):
        r"""
        Worker to run tracker on records

        Arguments
        ---------
        records:
            specific records, can be a subset of whole sequence
        dev: torch.device object
            target device
        result_queue:
            queue for result collecting
        speed_queue:
            queue for fps measurement collecting
        """
        #tracker = copy.deepcopy(self._pipeline)
        tracker = self.test_tracker
        for v_id, video in enumerate(records):
            lost, speed = self.track_single_video(tracker, video, v_id=v_id)
            if result_queue is not None:
                result_queue.put_nowait(lost)
            if speed_queue is not None:
                speed_queue.put_nowait(speed)

    def evaluation(self):
        for dataset_name in self.dataset_names:
            save_root_dir = os.path.join(self.result_root_dir,self.dataset_name)
            vot_root = self.data_root[self.dataset_name]
            vot_eval_show = vot_eval(dataset_name, vot_root, save_root_dir, num = 8)
            vot_eval_show.eval(self.test_name)

    def track_single_video(self, tracker, video, v_id=0):
        regions = []
        video = self.dataset[video]
        image_files, gt = video['image_files'], video['gt']
        start_frame, end_frame, lost_times, toc = 0, len(image_files), 0, 0
        for f, image_file in enumerate(tqdm(image_files)):
            ###############################注意 读取图片 有无更换通道位置
            # im = vot_benchmark.get_img(image_file)
            im = read_image(image_file)
            im_show = im.copy().astype(np.uint8)

            tic = cv2.getTickCount()
            if f == start_frame:  # init
                cx, cy, w, h = vot_benchmark.get_axis_aligned_bbox(gt[f])
                location = vot_benchmark.cxy_wh_2_rect((cx, cy), (w, h))  # x0, y0 , w, h  > left top
                tracker.init(im, gt[f], video['name'])
                regions.append(1 if 'VOT' in self.dataset_name else gt[f])
                gt_polygon = None
                pred_polygon = None
            elif f > start_frame:  # tracking

                location = tracker.update(im)

                gt_polygon = (gt[f][0], gt[f][1], gt[f][2], gt[f][3], gt[f][4],
                              gt[f][5], gt[f][6], gt[f][7])
                pred_polygon = (location[0], location[1],
                                location[0] + location[2], location[1],
                                location[0] + location[2],
                                location[1] + location[3], location[0],
                                location[1] + location[3])
                pred_polygon_outpoly = location

                b_overlap = vot_benchmark.vot_overlap(
                    gt_polygon, pred_polygon_outpoly, (im.shape[1], im.shape[0]))
               # print(str(b_overlap))


                if b_overlap:
                    regions.append(location)  # pred_polygon
                else:  # lost
                    regions.append(2)
                    lost_times += 1
                    start_frame = f + 5  # skip 5 frames
            else:  # skip
                regions.append(0)
            toc += cv2.getTickCount() - tic

        toc /= cv2.getTickFrequency()

        # save result
        result_dir = join(self.save_root_dir, video['name'])
        ensure_dir(result_dir)
        result_path = join(result_dir, '{:s}_001.txt'.format(video['name']))
        with open(result_path, "w") as fin:
            for x in regions:
                fin.write("{:d}\n".format(x)) if isinstance(x, int) else \
                    fin.write(','.join([vot_benchmark.vot_float2str("%.4f", i) for i in x]) + '\n')

        print(
            '({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps Lost: {:d} '
            .format(v_id, video['name'], toc, f / toc, lost_times))

        return lost_times, f / toc

