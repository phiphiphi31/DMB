from easydict import EasyDict 

OPTION = EasyDict()

# ------------------------------------------ data configuration ---------------------------------------------
OPTION.trainset = ['VOS'] # 'DAVIS17','Alivos' VOS
OPTION.valset = 'DAVIS16' # Alivos_test DAVIS17 DAVIS16
OPTION.datafreq = [5, 1]
OPTION.input_size = (384, 384)   # input image size (240, 427) (320, 480) 640, 960
OPTION.sampled_frames = 3        # min sampled time length while trianing
OPTION.max_skip = [5, 3]         # max skip time length while trianing
OPTION.samples_per_video = 2    # sample numbers per video

# ----------------------------------------- model configuration ---------------------------------------------
OPTION.keydim = 128
OPTION.valdim = 512
OPTION.save_freq = 20          #################################  5
OPTION.epochs_per_increment = 5

# ---------------------------------------- training configuration -------------------------------------------
OPTION.epochs = 80 #120
OPTION.train_batch = 4
OPTION.learning_rate = 0.00001 #0.00001
OPTION.gamma = 0.1
OPTION.momentum = (0.9, 0.999)
OPTION.solver = 'adam'             # 'sgd' or 'adam'
OPTION.weight_decay = 5e-4
OPTION.iter_size = 1
OPTION.milestone = []              # epochs to degrades the learning rate
OPTION.loss = 'both'               # 'ce' or 'iou' or 'both'
OPTION.mode = 'recurrent'          # 'mask' or 'recurrent' or 'threshold'
OPTION.iou_threshold = 0.65        # used only for 'threshold' training

# ---------------------------------------- testing configuration ------------ --------------------------------
OPTION.epoch_per_test = 1

# ------------------------------------------- other configuration -------------------------------------------
OPTION.checkpoint = '/home/jaffe/PycharmProjects/DMB/workspace/DMB_bz4'
OPTION.initial = '/home/jaffe/PycharmProjects/DMB/pytracking/networks/STM_weights.pth'#'/data2/jaffeProj/videoanalyst_vos/models/STM_weights.pth' #'/data2/jaffeProj/videoanalyst_vos/models/STM_weights.pth'    #'/data2/jaffeProj/videoanalyst_vos/models/STM_weights.pth'      # path to initialize the backbone
OPTION.resume = ''
OPTION.gpu_id = '0'      # defualt gpu-id (if not specified in cmd)
OPTION.workers = 32  # 4
OPTION.save_indexed_format = True # set True to save indexed format png file, otherwise segmentation with original image
OPTION.output_dir = 'DMB_bz4/val'
OPTION.weights = ''
