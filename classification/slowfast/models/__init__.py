#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .build import MODEL_REGISTRY, build_model  # noqa
from .custom_video_model_builder import *  # noqa
from .ptv_model_builder import (
    PTVCSN,
    PTVX3D,
    PTVR2plus1D,
    PTVResNet,
    PTVSlowFast,
)  # noqa
from .video_model_builder import ResNet, SlowFast  # noqa
from .uniformer import Uniformer  # noqa
from .s_vip_16 import SequentialVisionPermutator_16  # noqa
from .s_vip_32 import SequentialVisionPermutator_32  # noqa
# from .swin import Recognizer3D
from .sf import SF,SF2
