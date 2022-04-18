#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .ava_dataset import Ava  # noqa
from .build import DATASET_REGISTRY, build_dataset  # noqa
from .charades import Charades  # noqa
from .imagenet import Imagenet  # noqa
from .kinetics import Kinetics  # noqa
# from .ptv_datasets import Ptvcharades, Ptvkinetics, Ptvssv2  # noqa
from .sth import Sth  # noqa
from .ug2 import Ug2  # noqa
from .ug2_sparse import Ug2_sparse
from .ai_city_sparse import Ai_city_sparse  # noqa
from .ai_city_dense import Ai_city_dense  # noqa
from .ai_city_sparse_test import Ai_city_sparse_test 
from .ai_city_snipt import Ai_city_snipt 
