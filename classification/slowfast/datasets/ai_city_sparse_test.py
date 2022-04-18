#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import json
import numpy as np
import os
import random
from itertools import chain as chain
import torch
from torchvision import transforms
import torch.utils.data
from collections import defaultdict
from petrel_client.client import Client

import slowfast.utils.logging as logging

from . import utils as utils
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import create_random_augment

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Ai_city_sparse_test(torch.utils.data.Dataset):
    """
    ai_city video loader. Construct the Sth video loader,
    then sample clips from the videos. For training and validation, a single
    clip is randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Load ai_city data (frame paths, labels, etc. )
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries for reading frames from disk.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "test",
        ], "Split '{}' not supported for ai_city".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing ai_city {}...".format(mode))
        self._construct_loader()

        self.aug = False
        self.rand_erase = False
        
        if self.cfg.DATA.MC:
            conf_path = '/mnt/lustre/share_data/likunchang.vendor/competition/petreloss.conf'
            self.client = Client(conf_path)
        else:
            self.client = None

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        split = self.cfg.DATA.SPLIT
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR,
            self.cfg.DATA.LABEL_PATH_TEMPLATE,
        )
        tmp = [x.strip().split(',') for x in open(path_to_file)]
        self._path_to_videos = list()
        for item in tmp:
            path = item[0]
            num_frames = int(item[1])
            start_frame = int(item[2])
            # 0: path, 1: num_frames
            self._path_to_videos.append([path, num_frames, start_frame])

        # Extend self when self._num_clips > 1 (during testing).
        self._path_to_videos = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._path_to_videos]
            )
        )
        self._spatial_temporal_idx = list(
            chain.from_iterable(
                [
                    range(self._num_clips)
                    for _ in range(len(self._path_to_videos))
                ]
            )
        )
        logger.info(
            "ai_city dataloader constructed "
            " (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def get_seq_frames(self, index, temporal_sample_index):
        """
        Given the video index, return the list of sampled frame indexes.
        Args:
            index (int): the video index.
            temporal_sample_index (int): temporal sample index.
        Returns:
            seq (list): the indexes of frames of sampled from the video.
        """
        num_frames = self.cfg.DATA.NUM_FRAMES
        video_length = self._path_to_videos[index][1]
        start_frame = self._path_to_videos[index][2]

        seg_size = float(video_length - 1) / num_frames
        seq = []
        duration = seg_size / (self.cfg.TEST.NUM_ENSEMBLE_VIEWS + 1)
        for i in range(num_frames):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            frame_index = start + int(duration * (temporal_sample_index + 1))
            seq.append(frame_index + start_frame)
        return seq

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video frames can be fetched.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): the index of the video.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train"]:
            # -1 indicates random sampling.
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
        elif self.mode in ["val", "test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        if self.mode in ["test"]:
            seq = self.get_seq_frames(index, temporal_sample_index)
        else:
            seq = self.get_seq_frames(index, 0)

        path_template = os.path.join(
                            self.cfg.DATA.PATH_PREFIX,
                            self._path_to_videos[index][0],
                            self.cfg.DATA.IMAGE_TEMPLATE
                        )
        if self.client:
            frames = torch.as_tensor(
                utils.load_images_from_ceph(
                    [path_template.format(frame) for frame in seq],
                    self.client,
            ))
        else:
            frames = torch.as_tensor(
                utils.retry_load_images(
                    [path_template.format(frame) for frame in seq],
                    self._num_retries,
            ))
        
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

        frames = utils.pack_pathway_output(self.cfg, frames)

        num_frames = self._path_to_videos[index][1]
        start_frame = self._path_to_videos[index][2]

        return frames, index, num_frames, start_frame

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
