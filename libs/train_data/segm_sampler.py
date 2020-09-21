import random
import torch.utils.data
from libs.train_data.tensorlist import TensorDict
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

def no_processing(data):
    return data


class SegmSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. Each training sample is a
    tuple consisting of i) a train frame, used to obtain the modulation vector, and ii) a set of test frames on which
    the IoU prediction loss is calculated.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A 'train frame' is then sampled randomly from the sequence. Next, depending on the
    frame_sample_mode, the required number of test frames are sampled randomly, either  from the range
    [train_frame_id - max_gap, train_frame_id + max_gap] in the 'default' mode, or from [train_frame_id, train_frame_id + max_gap]
    in the 'causal' mode. Only the frames in which the target is visible are sampled, and if enough visible frames are
    not found, the 'max_gap' is incremented.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap, num_test_frames=1, processing=no_processing,
                 frame_sample_mode='default'):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train (reference) frame and the test frames.
            num_test_frames - Number of test frames used for calculating the IoU prediction loss.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'default' or 'causal'. If 'causal', then the test frames are sampled in a causal
                                manner.
        """
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [1 for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x/p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_test_frames = num_test_frames
        self.num_train_frames = 1                         # Only a single train frame allowed
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

        self.max_skip = 2
        self.increment = 1

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def increase_max_skip(self):
        MAX_TRAINING_SKIP = 100
        self.max_skip = min(self.max_skip + self.increment, MAX_TRAINING_SKIP)

    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        is_video_dataset = dataset.is_video_sequence()

        min_visible_frames = 2 * (self.num_test_frames + self.num_train_frames)
        enough_visible_frames = False

        # Sample a sequence with enough visible frames and get anno for the same
        while not enough_visible_frames:
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)
            anno, visible = dataset.get_sequence_info(seq_id)
            num_visible = visible.type(torch.int64).sum().item()
            enough_visible_frames = ((not is_video_dataset) and num_visible > 0) or (num_visible > min_visible_frames and len(visible) >= 20)

        if is_video_dataset:
            train_frame_ids = None
            test_frame1_ids = None
            test_frame2_ids = None
            gap_increase = 0

            #Sample frame numbers
            while (test_frame1_ids is  None) or (test_frame2_ids is None):
                train_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames)
                test_frame1_ids = self._sample_visible_ids(visible, min_id=train_frame_ids[0] + 1 ,
                                                              max_id=train_frame_ids[0] + self.max_skip ,
                                                              num_ids=self.num_test_frames)
                if test_frame1_ids is not None:
                   test_frame2_ids = self._sample_visible_ids(visible, min_id=test_frame1_ids[0] + 1 ,
                                                                       max_id=test_frame1_ids[0] + self.max_skip,
                                                                       num_ids=self.num_test_frames)

        # Get frames

        if 'VOS' in dataset.get_name():
            # Prepare data
            train_frames, train_masks, train_anno, object_meta = dataset.get_frames(seq_id, train_frame_ids, anno)
            test1_frames, test1_masks, test1_anno, object_meta1 = dataset.get_frames(seq_id, test_frame1_ids, anno)
            test2_frames, test2_masks, test2_anno, object_meta2 = dataset.get_frames(seq_id, test_frame2_ids, anno)

            data = TensorDict({'train_images': train_frames, # list[1]: np[720, 1280, 3]
                               'train_anno': train_anno, # list[1]: tensor[4]
                               'train_masks': train_masks,  # list[1]: np[720, 1280, 1]

                               'test1_images': test1_frames, # list[1]: np[720, 1280, 3]
                               'test1_anno': test1_anno,  # list[1]: tensor[4]
                               'test1_masks': test1_masks,

                               'test2_images': test2_frames,  # list[1]: np[720, 1280, 3]
                               'test2_anno': test2_anno,  # list[1]: tensor[4]
                               'test2_masks': test2_masks,

                               'dataset': dataset.get_name(),  # 'VOS'
                               }) # list[1]: np[720, 1280, 1]
            """
            im1 = data['train_images'][0]
            cv2.imwrite('/data2/jaffeProj/debug/img1.png', im1)

            im2 = data['test1_images'][0]
            cv2.imwrite('/data2/jaffeProj/debug/img2.png', im2)

            im3 = data['test2_images'][0]
            cv2.imwrite('/data2/jaffeProj/debug/img3.png', im3)

            show1 = data['train_masks'][0][:,:, 0]
            sns.heatmap(show1)
            plt.show()

            show2 = data['test1_masks'][0][:, :, 0]
            sns.heatmap(show2)
            plt.show()

            show3 = data['test2_masks'][0][:, :, 0]
            sns.heatmap(show3)
            plt.show()
            
            """


            return self.processing(data)

