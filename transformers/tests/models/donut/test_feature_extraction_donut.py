# coding=utf-8
# Copyright 2022 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest

import numpy as np

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_feature_extraction_common import (
    FeatureExtractionSavingTestMixin,
    prepare_image_inputs,
)


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

    from transformers import DonutFeatureExtractor


class DonutFeatureExtractionTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=[20, 18],
        do_thumbnail=True,
        do_align_axis=False,
        do_pad=True,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.do_thumbnail = do_thumbnail
        self.do_align_axis = do_align_axis
        self.do_pad = do_pad
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

    def prepare_feat_extract_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_thumbnail": self.do_thumbnail,
            "do_align_long_axis": self.do_align_axis,
            "do_pad": self.do_pad,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
        }


@require_torch
@require_vision
class DonutFeatureExtractionTest(FeatureExtractionSavingTestMixin, unittest.TestCase):

    feature_extraction_class = DonutFeatureExtractor if is_vision_available() else None

    def setUp(self):
        self.feature_extract_tester = DonutFeatureExtractionTester(self)

    @property
    def feat_extract_dict(self):
        return self.feature_extract_tester.prepare_feat_extract_dict()

    def test_feat_extract_properties(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        self.assertTrue(hasattr(feature_extractor, "do_resize"))
        self.assertTrue(hasattr(feature_extractor, "size"))
        self.assertTrue(hasattr(feature_extractor, "do_thumbnail"))
        self.assertTrue(hasattr(feature_extractor, "do_align_long_axis"))
        self.assertTrue(hasattr(feature_extractor, "do_pad"))
        self.assertTrue(hasattr(feature_extractor, "do_normalize"))
        self.assertTrue(hasattr(feature_extractor, "image_mean"))
        self.assertTrue(hasattr(feature_extractor, "image_std"))

    def test_batch_feature(self):
        pass

    def test_call_pil(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        # create random PIL images
        image_inputs = prepare_image_inputs(
            self.feature_extract_tester, equal_resolution=False
        )
        for image in image_inputs:
            self.assertIsInstance(image, Image.Image)

        # Test not batched input
        encoded_images = feature_extractor(
            image_inputs[0], return_tensors="pt"
        ).pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                1,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.size[1],
                self.feature_extract_tester.size[0],
            ),
        )

        # Test batched
        encoded_images = feature_extractor(
            image_inputs, return_tensors="pt"
        ).pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.size[1],
                self.feature_extract_tester.size[0],
            ),
        )

    def test_call_numpy(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        # create random numpy tensors
        image_inputs = prepare_image_inputs(
            self.feature_extract_tester, equal_resolution=False, numpify=True
        )
        for image in image_inputs:
            self.assertIsInstance(image, np.ndarray)

        # Test not batched input
        encoded_images = feature_extractor(
            image_inputs[0], return_tensors="pt"
        ).pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                1,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.size[1],
                self.feature_extract_tester.size[0],
            ),
        )

        # Test batched
        encoded_images = feature_extractor(
            image_inputs, return_tensors="pt"
        ).pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.size[1],
                self.feature_extract_tester.size[0],
            ),
        )

    def test_call_pytorch(self):
        # Initialize feature_extractor
        feature_extractor = self.feature_extraction_class(**self.feat_extract_dict)
        # create random PyTorch tensors
        image_inputs = prepare_image_inputs(
            self.feature_extract_tester, equal_resolution=False, torchify=True
        )
        for image in image_inputs:
            self.assertIsInstance(image, torch.Tensor)

        # Test not batched input
        encoded_images = feature_extractor(
            image_inputs[0], return_tensors="pt"
        ).pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                1,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.size[1],
                self.feature_extract_tester.size[0],
            ),
        )

        # Test batched
        encoded_images = feature_extractor(
            image_inputs, return_tensors="pt"
        ).pixel_values
        self.assertEqual(
            encoded_images.shape,
            (
                self.feature_extract_tester.batch_size,
                self.feature_extract_tester.num_channels,
                self.feature_extract_tester.size[1],
                self.feature_extract_tester.size[0],
            ),
        )
