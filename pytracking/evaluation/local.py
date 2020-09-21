from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.got10k_path = '/home/jaffe/Titanspace/datasets/dataset/GOT-10k'
    settings.lasot_path = '/home/jaffe/Titanspace/datasets/dataset/LaSOT/LaSOTBenchmark'
    settings.mobiface_path = ''
    settings.network_path = '/home/jaffe/PycharmProjects/d3s_me/pytracking/networks'    # Where tracking networks are stored.
   # settings.network_path ='/home/jaffe/PycharmProjects/d3s_me/NoSet3_lr103_PreMask_Segm_GAN/checkpoints/ltr/segmRA_GAN/segmRA_GAN'
    settings.nfs_path = ''
    settings.otb_path = '/home/jaffe/Titanspace/datasets/dataset/testdataset/dataset/OTB100'
    settings.results_path = '/home/jaffe/PycharmProjects/d3s_me/pytracking/tracking_results'    # Where to store tracking results
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/jaffe/Titanspace/datasets/dataset/TrackingNet'
    settings.uav_path = ''
    settings.vot16_path = ''
    settings.vot18_path = '/home/jaffe/dataset/VOT/vot2019/VOT2019'
    settings.vot_path = '/home/jaffe/dataset/VOT/vot2018/VOT2018'
    return settings

