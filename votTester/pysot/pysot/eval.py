import os
import sys
import time
import argparse
import functools

from pytracking.pysot.datasets import OTBDataset, LaSOTDataset, UAVDataset, NFSDataset, VOTDataset, VOTLTDataset
from pytracking.pysot.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, EAOBenchmark, F1Benchmark
from pytracking.pysot.visualization import draw_success_precision, draw_f1

sys.path.append("./")

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
# from .datasets import OTBDataset, UAVDataset, LaSOTDataset, VOTDataset, NFSDataset, VOTLTDataset
# from .evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, EAOBenchmark, F1Benchmark
# from .visualization import draw_success_precision, draw_eao, draw_f1

def eval_meter(result_list = None,dataset = 'VOT2018',show_level=0,tracker_result_dir=None):

    if dataset == 'OTB100':
        data_dir = '/media/adminer/data/OTB-pysot-toolkit'
    if dataset == 'VOT2018':
        data_dir = "/home/jaffe/dataset/VOT/vot2018/VOT2018"
    if dataset == 'VOT2019':
        data_dir = "/media/adminer/data/VOT19/VOT2019/"
    if dataset == 'LaSOT':
        data_dir = "/home/adminer/Desktop/WANGNING/Dataset/LaSOT/LaSOTBenchmark/"

    data_name = dataset
    trackers_list = [] + result_list



    parser = argparse.ArgumentParser(description='Single Object Tracking Evaluation')
    parser.add_argument('--dataset_dir', type=str,default=data_dir, help='dataset root directory')
    parser.add_argument('--dataset',  type=str,default=data_name, help='dataset name')
    parser.add_argument('--tracker_result_dir', type=str,default=tracker_result_dir, help='tracker result root')
    parser.add_argument('--trackers',default=trackers_list)
    parser.add_argument('--vis',dest='vis', action='store_true')
    parser.add_argument('--show_video_level',default=show_level,dest='show_video_level', action='store_true')
    parser.add_argument('--num', type=int, help='number of processes to eval', default=1)
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


if __name__ == '__main__':

    tracker_result_dir ="/home/jaffe/PycharmProjects/d3s_me/votTester/results/vot2018"
    eval_meter(result_list=['d3s_train3',
                            'd3s_train2',
                            ],dataset='VOT2018',tracker_result_dir=tracker_result_dir)