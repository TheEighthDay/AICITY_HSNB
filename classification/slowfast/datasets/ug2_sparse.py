#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr
from torchvision import transforms
from petrel_client.client import Client
import numpy as np
from glob import glob 
from PIL import Image
import torchvision
from torchvision.transforms.functional import adjust_gamma

import slowfast.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import create_random_augment

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Ug2_sparse(torch.utils.data.Dataset):
    """
    Ug2 video loader. Construct the Ug2 video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10, decording=None):
        """
        Construct the Ug2 video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
            decording: deocord mode [decord decord_GIC decord_AdaptGIC frame]
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train_label",
            "train_unlabel",
            "val_label",
            "val_dry",
            "val_aridv1.5",
            "test_aridv1.5",
            "pseudo_train_unlabel",
        ], "Split '{}' not supported for Ug2".format(mode)
        self.mode = mode
        self.cfg = cfg
        if decording is None:
            self.decording = cfg.DATA.DECODING_BACKEND
        else:
            self.decording = decording

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ['train_label','train_unlabel','val_label','val_dry','val_aridv1.5','pseudo_train_unlabel']:
            self._num_clips = 1
            cfg.TEST.NUM_ENSEMBLE_VIEWS = 1
            cfg.TEST.NUM_SPATIAL_CROPS = 1
        elif self.mode in ["test_aridv1.5"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing Ug2 {}...".format(mode))
        logger.info("Number of clips {}".format(self._num_clips))
        self._construct_loader()
        self.aug = False
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0

        if self.mode in ['train_label','train_unlabel'] and self.cfg.AUG.ENABLE:
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

        if self.cfg.DATA.MC:
            conf_path = '/mnt/lustre/likunchang.vendor/petreloss.conf'
            self.client = Client(conf_path)
        else:
            self.client = None

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        # set the path manually
        # /mnt/lustre/share_data/likunchang.vendor/ug2_dataset
        if self.mode == 'train_label':
            path_to_file = os.path.join(self.cfg.DATA.PATH_PREFIX, "train_label.csv")
            video_prefix = os.path.join(self.cfg.DATA.PATH_PREFIX, "labeled_train/Train")
        elif self.mode == 'train_unlabel':
            path_to_file = os.path.join(self.cfg.DATA.PATH_PREFIX, "train_unlabel.csv")
            video_prefix = os.path.join(self.cfg.DATA.PATH_PREFIX, "dark_train/Train")
        elif self.mode == 'pseudo_train_unlabel':
            path_to_file = os.path.join(self.cfg.DATA.PATH_PREFIX, "train_unlabel.csv")
            video_prefix = os.path.join(self.cfg.DATA.PATH_PREFIX, "dark_train/Train")
        elif self.mode == 'val_label':
            path_to_file = os.path.join(self.cfg.DATA.PATH_PREFIX, "train_label.csv")
            video_prefix = os.path.join(self.cfg.DATA.PATH_PREFIX, "labeled_train/Train")
        elif self.mode == 'val_dry':
            path_to_file = os.path.join(self.cfg.DATA.PATH_PREFIX, "val_dry.csv")
            video_prefix = os.path.join(self.cfg.DATA.PATH_PREFIX, "dry_run/Validation")
        elif self.mode == 'val_aridv1.5':
            path_to_file = os.path.join(self.cfg.DATA.PATH_PREFIX, "val_aridv1.5.csv")
            video_prefix = os.path.join(self.cfg.DATA.PATH_PREFIX, "ARID_v1.5/clips_v1.5")
        elif self.mode == 'test_aridv1.5':
            path_to_file = os.path.join(self.cfg.DATA.PATH_PREFIX, "test_aridv1.5.csv")
            video_prefix = os.path.join(self.cfg.DATA.PATH_PREFIX, "ARID_v1.5/clips_v1.5")
        else:
            Exception("Bug of mode in dataset{}".format(self.mode))

        assert g_pathmgr.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with g_pathmgr.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert (
                    len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
                    == 2
                )
                path, label = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(video_prefix, path)
                    )
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Ug2 split {} from {}".format(
            self._split_idx, path_to_file
        )
        logger.info(
            "Constructing Ug2 dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def get_seq_frames(self, video_length, temporal_sample_index):
        """
        Given the video length, return the list of sampled frame indexes.
        Args:
            video_length (int): the video length.
            temporal_sample_index (int): temporal sample index.
        Returns:
            seq (list): the indexes of frames of sampled from the video.
        """
        num_frames = self.cfg.DATA.NUM_FRAMES

        seg_size = float(video_length - 1) / num_frames
        seq = []
        # index from 1, must add 1
        if self.mode in ['train_label','train_unlabel']:
            for i in range(num_frames):
                start = int(np.round(seg_size * i))
                end = int(np.round(seg_size * (i + 1)))
                seq.append(random.randint(start, end))
        else:
            duration = seg_size / (self.cfg.TEST.NUM_ENSEMBLE_VIEWS + 1)
            for i in range(num_frames):
                start = int(np.round(seg_size * i))
                end = int(np.round(seg_size * (i + 1)))
                frame_index = start + int(duration * (temporal_sample_index + 1))
                seq.append(frame_index )
        return seq

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ['train_label','train_unlabel']:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ['val_label','val_dry','val_aridv1.5','pseudo_train_unlabel','test_aridv1.5']:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )

            # no jitter and resize are performed
            min_scale, max_scale, crop_size = ([self.cfg.DATA.TEST_CROP_SIZE] * 3)

        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None
            if self.decording == "decord" or self.decording == "decord_GIC" or self.decording == "decord_AdaptGIC":
                try:
                    video_container = container.get_video_container(
                        self.client,
                        self._path_to_videos[index],
                        self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                        self.decording,
                    )
                except Exception as e:
                    logger.info(
                        "Failed to load video from {} with error {}".format(
                            self._path_to_videos[index], e
                        )
                    )
                # Select a random video if the current video was not able to access.
                if video_container is None:
                    logger.warning(
                        "Failed to meta load video idx {} from {}; trial {}".format(
                            index, self._path_to_videos[index], i_try
                        )
                    )
                    if self.mode not in ['test_aridv1.5'] and i_try > self._num_retries // 2:
                        # let's try another one
                        index = random.randint(0, len(self._path_to_videos) - 1)
                    continue
                
                video_length = len(video_container)
                if self.mode in ['test_aridv1.5']:
                    seq = self.get_seq_frames(video_length, temporal_sample_index)
                else:
                    seq = self.get_seq_frames(video_length, 0)

                # Decode video. Meta info is used to perform selective decoding.
                if self.decording == "decord":
                    tmp_frames = [video_container[i] for i in seq]
                elif self.decording == "decord_GIC":
                    tmp_frames = [video_container[i] for i in seq]
                    gamma_val = 0.45
                    for i in range(len(tmp_frames)):
                        GIC = adjust_gamma(tmp_frames[i].permute(2,0,1),gamma_val).permute(1,2,0)
                        tmp_frames[i] = GIC
                else:
                    tmp_frames = [video_container[i] for i in seq]
                    gamma_val = 0.45
                    for i in range(len(tmp_frames)):
                        t = tmp_frames[i].permute(2,0,1)
                        R = t[0]
                        G = t[1]
                        B = t[2]
                        k=0.299*R+0.587*G+0.114*B
                        if torch.mean(k) > 0:
                            gamma_val = math.log10(0.5) / math.log10(torch.mean(k) / 255)
                        else:
                            gamma_val = 0.45
                        GIC = adjust_gamma(tmp_frames[i].permute(2,0,1),gamma_val).permute(1,2,0)
                        tmp_frames[i] = GIC
                
                frames = torch.stack(tmp_frames)
                
            elif self.decording == "frame":
                frame_path = os.path.join(self.cfg.DATA.FRAME_PATH,self._path_to_videos[index].split("/")[-1].split(".")[0])
                glob_frame_path = glob(frame_path + '/*.*')
                video_length = len(glob_frame_path)

                if self.mode in ['test_aridv1.5']:
                    seq = self.get_seq_frames(video_length, temporal_sample_index)
                else:
                    seq = self.get_seq_frames(video_length, 0)
                # tmp_frames = [video_container[i] for i in seq]
                index = seq

                frame_name = frame_path + "/" + frame_path.split('/')[-1] + "_{}".format(index[0])+".png"
                # print("frame_name"frame_name)
                frame_image = torchvision.transforms.functional.to_tensor(Image.open(frame_name)).permute(1,2,0).unsqueeze(0)
                # print("frame_image",frame_image.size())
                for i in index[1:]:
                    frame_name = frame_path+ "/"  + frame_path.split('/')[-1] + "_{}".format(i)+".png"
                    frame_image = torch.cat([torchvision.transforms.functional.to_tensor(Image.open(frame_name)).permute(1,2,0).unsqueeze(0),frame_image],dim=0)
                    # print("frame_image",frame_image.size())
                # print("frames[0]",type(frames[0]),frames[0].size())
                # print("---------index",index)
                # tmp_frames = [frames[i.item()] for i in index]
                frames = frame_image
            else:
                Exception("Decord backbone error")

            

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                logger.warning(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ['test_aridv1.5'] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            if self.aug:
                if self.cfg.AUG.NUM_SAMPLE > 1:

                    frame_list = []
                    label_list = []
                    index_list = []
                    for _ in range(self.cfg.AUG.NUM_SAMPLE):
                        new_frames = self._aug_frame(
                            frames,
                            spatial_sample_index,
                            min_scale,
                            max_scale,
                            crop_size,
                        )
                        label = self._labels[index]
                        new_frames = utils.pack_pathway_output(
                            self.cfg, new_frames
                        )
                        frame_list.append(new_frames)
                        label_list.append(label)
                        index_list.append(index)
                    return frame_list, label_list, index_list, {}

                else:
                    frames = self._aug_frame(
                        frames,
                        spatial_sample_index,
                        min_scale,
                        max_scale,
                        crop_size,
                    )

            else:
                frames = utils.tensor_normalize(
                    frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                )
                # T H W C -> C T H W.
                frames = frames.permute(3, 0, 1, 2)
                # Perform data augmentation.
                frames = utils.spatial_sampling(
                    frames,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                )

            label = self._labels[index]
            frames = utils.pack_pathway_output(self.cfg, frames)
            return frames, label, index, {}
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def _aug_frame(
        self,
        frames,
        spatial_sample_index,
        min_scale,
        max_scale,
        crop_size,
    ):
        aug_transform = create_random_augment(
            input_size=(frames.size(1), frames.size(2)),
            auto_augment=self.cfg.AUG.AA_TYPE,
            interpolation=self.cfg.AUG.INTERPOLATION,
        )
        # T H W C -> T C H W.
        frames = frames.permute(0, 3, 1, 2)
        list_img = self._frame_to_list_img(frames)
        list_img = aug_transform(list_img)
        frames = self._list_img_to_frames(list_img)
        frames = frames.permute(0, 2, 3, 1)

        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
            self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
        )
        relative_scales = (
            None if (self.mode not in ['train_label','train_unlabel'] or len(scl) == 0) else scl
        )
        relative_aspect = (
            None if (self.mode not in ['train_label','train_unlabel'] or len(asp) == 0) else asp
        )
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
            if self.mode in ['train_label','train_unlabel']
            else False,
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                self.cfg.AUG.RE_PROB,
                mode=self.cfg.AUG.RE_MODE,
                max_count=self.cfg.AUG.RE_COUNT,
                num_splits=self.cfg.AUG.RE_COUNT,
                device="cpu",
            )
            frames = frames.permute(1, 0, 2, 3)
            frames = erase_transform(frames)
            frames = frames.permute(1, 0, 2, 3)

        return frames

    def _frame_to_list_img(self, frames):
        img_list = [
            transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))
        ]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
