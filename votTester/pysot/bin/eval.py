import os
import sys
import time
import argparse
import functools

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from votTester.pysot.pysot.datasets import OTBDataset, UAVDataset, LaSOTDataset, VOTDataset, NFSDataset, VOTLTDataset
from votTester.pysot.pysot.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, EAOBenchmark, F1Benchmark
from votTester.pysot.pysot.visualization import draw_success_precision, draw_eao, draw_f1

class vot_eval:
    def __init__(self, dataset, dataset_dir, tracker_result_dir, num = 8 ):
        self.dataset = dataset
        self.trackers = []
        self.dataset_dir = dataset_dir
        self.num = num
        self.tracker_dir = tracker_result_dir

        self.args_num = min(self.num, 1)

    def eval(self, tracker):
        self.trackers.append(tracker)

        if 'VOT2018' == self.dataset or 'VOT2016' == self.dataset or 'VOT2019' == self.dataset\
            or 'vot2018' == self.dataset or 'vot2016' == self.dataset or 'vot2019' == self.dataset:
            dataset = VOTDataset(self.dataset, self.dataset_dir)
            dataset.set_tracker(self.tracker_dir, self.trackers)
            ar_benchmark = AccuracyRobustnessBenchmark(dataset)
            ar_result = {}
            with Pool(processes=self.num) as pool:
                for ret in tqdm(pool.imap_unordered(ar_benchmark.eval,
                    self.trackers), desc='eval ar', total=len(self.trackers), ncols=100):
                    ar_result.update(ret)
                    # benchmark.show_result(ar_result)

            benchmark = EAOBenchmark(dataset)
            eao_result = {}
            with Pool(processes=self.args_num) as pool:
                for ret in tqdm(pool.imap_unordered(benchmark.eval,
                    self.trackers), desc='eval eao', total=len(self.trackers), ncols=100):
                    eao_result.update(ret)
            # benchmark.show_result(eao_result)
            ar_benchmark.show_result(ar_result, eao_result,
                    show_video_level = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single Object Tracking Evaluation')
    parser.add_argument('--dataset_dir', type=str,  default = '/home/jaffe/Titanspace/datasets/dataset/testdataset/dataset/VOT/vot2018/VOT2018', help='dataset root directory')
    parser.add_argument('--dataset', type=str, default = 'VOT2018', help='dataset name')
    parser.add_argument('--tracker_result_dir', default='/home/jaffe/PycharmProjects/d3s_me/new_stm', type=str, help='tracker result root')
    parser.add_argument('--trackers', default = [
                                                 'recurrent1.pth.tarresult',
                                                 # 'recurrent32.pth.tarresult',
                                                 #'recurrent33.pth.tarresult',
                                                 # recurrent34.pth.tarresult',
                                                 #'recurrent35.pth.tarresult',
                                                 #'recurrent36.pth.tarresult',
                                                 #'recurrent37.pth.tarresult',
                                                 #'recurrent38.pth.tarresult',
                                                 #'recurrent39.pth.tarresult',
                                                 'recurrent40.pth.tarresult',
                                                 'recurrent10.pth.tarresult',
                                                 'recurrent2.pth.tarresult',
                                                 'recurrent3.pth.tarresult',
                                                 'recurrent4.pth.tarresult',
                                                 'recurrent5.pth.tarresult',
                                                 'recurrent6.pth.tarresult',
                                                 'recurrent7.pth.tarresult',
                                                 'recurrent8.pth.tarresult',
                                                 'recurrent9.pth.tarresult',
                                            #     'recurrent40.pth.tarresult'
    ], nargs='+')




    parser.add_argument('--vis', dest='vis', action='store_true')
    parser.add_argument('--show_video_level', dest='show_video_level', action='store_true')
    parser.add_argument('--num', type=int, help='number of processes to eval', default=8)
    args = parser.parse_args()

    tracker_dir = args.tracker_result_dir
    trackers = args.trackers
    root = args.dataset_dir

    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    if 'OTB' in args.dataset:
        dataset = OTBDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                            name=dataset.name,
                            videos=videos,
                            attr=attr,
                            precision_ret=precision_ret)
    elif 'LaSOT' == args.dataset:
        dataset = LaSOTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        # success_ret = benchmark.eval_success(trackers)
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        norm_precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_norm_precision,
                trackers), desc='eval norm precision', total=len(trackers), ncols=100):
                norm_precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret, norm_precision_ret,
                show_video_level=args.show_video_level)
        if args.vis:
            draw_success_precision(success_ret,
                    name=dataset.name,
                    videos=dataset.attr['ALL'],
                    attr='ALL',
                    precision_ret=precision_ret,
                    norm_precision_ret=norm_precision_ret)
    elif 'UAV' in args.dataset:
        dataset = UAVDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                        name=dataset.name,
                        videos=videos,
                        attr=attr,
                        precision_ret=precision_ret)
    elif 'NFS' in args.dataset:
        dataset = NFSDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                            name=dataset.name,
                            video=videos,
                            attr=attr,
                            precision_ret=precision_ret)
    elif 'VOT2018' == args.dataset or 'VOT2016' == args.dataset or 'VOT2019' == args.dataset:
        dataset = VOTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        ar_benchmark = AccuracyRobustnessBenchmark(dataset)
        ar_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(ar_benchmark.eval,
                trackers), desc='eval ar', total=len(trackers), ncols=100):
                ar_result.update(ret)
        # benchmark.show_result(ar_result)

        benchmark = EAOBenchmark(dataset)
        eao_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                trackers), desc='eval eao', total=len(trackers), ncols=100):
                eao_result.update(ret)
        # benchmark.show_result(eao_result)
        ar_benchmark.show_result(ar_result, eao_result,
                show_video_level=args.show_video_level)
    elif 'VOT2018-LT' == args.dataset:
        dataset = VOTLTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = F1Benchmark(dataset)
        f1_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                trackers), desc='eval f1', total=len(trackers), ncols=100):
                f1_result.update(ret)
        benchmark.show_result(f1_result,
                show_video_level=args.show_video_level)
        if args.vis:
            draw_f1(f1_result)




