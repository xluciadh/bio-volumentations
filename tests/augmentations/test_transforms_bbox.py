# ============================================================================================= #
#  Author:       Jakub Polonský, Lucia Hradecká                                                 #
#  Copyright:    Jakub Polonský                                                                 #
#                Lucia Hradecká     : lucia.d.hradecka@gmail.com                                #
#                                                                                               #
#  MIT License.                                                                                 #
#                                                                                               #
#  Permission is hereby granted, free of charge, to any person obtaining a copy                 #
#  of this software and associated documentation files (the "Software"), to deal                #
#  in the Software without restriction, including without limitation the rights                 #
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell                    #
#  copies of the Software, and to permit persons to whom the Software is                        #
#  furnished to do so, subject to the following conditions:                                     #
#                                                                                               #
#  The above copyright notice and this permission notice shall be included in all               #
#  copies or substantial portions of the Software.                                              #
#                                                                                               #
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR                   #
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,                     #
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE                  #
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER                       #
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,                #
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE                #
#  SOFTWARE.                                                                                    #
# ============================================================================================= #


import unittest

import numpy as np

from src.bio_volumentations.core.composition import Compose
from src.bio_volumentations.augmentations.transforms import AffineTransform, CenterCrop, Flip, RandomRotate90, Resize, \
    Rescale, Pad, Scale
from src.bio_volumentations.core.transforms_interface import DualTransform
from src.bio_volumentations.augmentations.functional_bbox import BoundingBox


#########################################################################
#                                                                       #
#                           Bbox class tests                            #
#                                                                       #
#########################################################################


class TestBBoxClass(unittest.TestCase):
    def test1_bbox_data(self):
        # define bbox data
        min_pt = (1, 0, 2)
        max_pt = (13, 12, 10)
        tp = 2
        class_label = 'cell nucleus'
        format_str = 'voc'

        bbox_data = {'min_pt': np.asarray(min_pt), 'max_pt': np.asarray(max_pt),
                     'time_pt': tp, 'class': class_label, 'format': format_str}

        # test: we can create the bbox
        bbox = BoundingBox(min_point=bbox_data['min_pt'], max_point=bbox_data['max_pt'],
                           time_point=bbox_data['time_pt'], class_label=bbox_data['class'],
                           bbox_format=bbox_data['format'])

        # test: the bbox stores its data
        self.assertTrue(np.allclose(bbox.min, min_pt))
        self.assertTrue(np.allclose(bbox.max, max_pt))
        self.assertEqual(bbox.time_point, tp)
        self.assertEqual(bbox.class_label, class_label)
        self.assertEqual(bbox.bbox_format, format_str)

        # test: compose without any transforms produces the same bbox with all data
        bbox_input_format = [min_pt, max_pt, tp, class_label]
        pipeline = Compose([])
        res_bbox_list = pipeline(image=np.zeros((20, 20, 20)), bboxes=[bbox_input_format], bbox_format=format_str)[
            'bboxes']
        self.assertIsInstance(res_bbox_list, list)
        self.assertEqual(len(res_bbox_list), 1)
        res_bbox = res_bbox_list[0]
        self.assertEqual(len(res_bbox), 4)
        self.assertTupleEqual(res_bbox[0], min_pt)
        self.assertTupleEqual(res_bbox[1], max_pt)
        self.assertEqual(res_bbox[2], tp)
        self.assertEqual(res_bbox[3], class_label)


#########################################################################
#                                                                       #
#                         Transformation tests                          #
#                                                                       #
#########################################################################


def get_bbox_transform_result(transform: DualTransform,
                              data: list[list[int]],
                              shape: tuple[int, int, int] = (18, 17, 16),
                              params: dict = None):
    d, h, w = shape
    sample = {
        'image': np.zeros((4, d, h, w)),
        'bboxes': data
    }

    if params is None:
        params = {}

    transformation = Compose([transform(**params, p=1)])
    transformation_res = transformation(**sample)

    return transformation_res['bboxes']


def are_lists_close(a, b, atol=1e-5):
    return len(a) == len(b) and np.allclose([[*(bbox[0]), *(bbox[1]), bbox[2]] for bbox in a],
                                            [[*(bbox[0]), *(bbox[1]), bbox[2]] for bbox in b],  # rtol=1e-01
                                            atol=atol)


def get_bbox_test_set():
    return [
        [(0, 0, 0), (17, 16, 15), 0],
        [(0, 0, 0), (17, 16, 15), 0],
        [(0, 0, 0), (8, 8, 8), 0],
        [(2, 3, 5), (13, 12, 11), 0],
        [(0, 0, 0), (1, 1, 1), 0],
    ]


class TestFlipBBox(unittest.TestCase):
    def test1_flip_all_axes(self):
        test = get_bbox_transform_result(Flip, get_bbox_test_set())
        correct = [
            [(0, 0, 0), (17, 16, 15), 0],
            [(0, 0, 0), (17, 16, 15), 0],
            [(9, 8, 7), (17, 16, 15), 0],
            [(4, 4, 4), (15, 13, 10), 0],
            [(16, 15, 14), (17, 16, 15), 0],
        ]

        self.assertListEqual(test, correct)

    def test2_flip_axis_1(self):
        test = get_bbox_transform_result(Flip, get_bbox_test_set(), params={'axes': [1]})
        correct = [
            [(0, 0, 0), (17, 16, 15), 0],
            [(0, 0, 0), (17, 16, 15), 0],
            [(9, 0, 0), (17, 8, 8), 0],
            [(4, 3, 5), (15, 12, 11), 0],
            [(16, 0, 0), (17, 1, 1), 0],
        ]

        self.assertListEqual(test, correct)

    def test2_flip_axis_2(self):
        test = get_bbox_transform_result(Flip, get_bbox_test_set(), params={'axes': [2]})
        correct = [
            [(0, 0, 0), (17, 16, 15), 0],
            [(0, 0, 0), (17, 16, 15), 0],
            [(0, 8, 0), (8, 16, 8), 0],
            [(2, 4, 5), (13, 13, 11), 0],
            [(0, 15, 0), (1, 16, 1), 0],
        ]

        self.assertListEqual(test, correct)

    def test2_flip_axis_3(self):
        test = get_bbox_transform_result(Flip, get_bbox_test_set(), params={'axes': [3]})
        correct = [
            [(0, 0, 0), (17, 16, 15), 0],
            [(0, 0, 0), (17, 16, 15), 0],
            [(0, 0, 7), (8, 8, 15), 0],
            [(2, 3, 4), (13, 12, 10), 0],
            [(0, 0, 14), (1, 1, 15), 0],
        ]

        self.assertListEqual(test, correct)


class TestRandomRotate90BBox(unittest.TestCase):
    def test1_axis_1(self):
        test = get_bbox_transform_result(RandomRotate90, get_bbox_test_set(), params={'factor': 1, 'axes': [1]})
        correct = [
            [(0, 0, 0), (17, 15, 16), 0],
            [(0, 0, 0), (17, 15, 16), 0],
            [(0, 0, 8), (8, 8, 16), 0],
            [(2, 5, 4), (13, 11, 13), 0],
            [(0, 0, 15), (1, 1, 16), 0],
        ]

        self.assertListEqual(test, correct)

    def test1_axis_2(self):
        test = get_bbox_transform_result(RandomRotate90, get_bbox_test_set(), params={'factor': 1, 'axes': [2]})
        correct = [
            [(0, 0, 0), (15, 16, 17), 0],
            [(0, 0, 0), (15, 16, 17), 0],
            [(7, 0, 0), (15, 8, 8), 0],
            [(4, 3, 2), (10, 12, 13), 0],
            [(14, 0, 0), (15, 1, 1), 0],
        ]

        self.assertListEqual(test, correct)

    def test1_axis_3(self):
        test = get_bbox_transform_result(RandomRotate90, get_bbox_test_set(), params={'factor': 1, 'axes': [3]})
        correct = [
            [(0, 0, 0), (16, 17, 15), 0],
            [(0, 0, 0), (16, 17, 15), 0],
            [(0, 9, 0), (8, 17, 8), 0],
            [(3, 4, 5), (12, 15, 11), 0],
            [(0, 16, 0), (1, 17, 1), 0],
        ]

        self.assertListEqual(test, correct)

    def test1_axis_3_2(self):
        test = get_bbox_transform_result(RandomRotate90, get_bbox_test_set(), params={'factor': 1, 'axes': [3, 2]})
        correct = [
            [(0, 0, 0), (15, 17, 16), 0],
            [(0, 0, 0), (15, 17, 16), 0],
            [(7, 9, 0), (15, 17, 8), 0],
            [(4, 4, 3), (10, 15, 12), 0],
            [(14, 16, 0), (15, 17, 1), 0],
        ]

        self.assertListEqual(test, correct)

    def test2_all_axes_to_identity(self):
        test = get_bbox_transform_result(RandomRotate90, get_bbox_test_set(), params={'factor': 2, 'axes': [1, 2, 3]})
        correct = [
            [(0, 0, 0), (17, 16, 15), 0],
            [(0, 0, 0), (17, 16, 15), 0],
            [(0, 0, 0), (8, 8, 8), 0],
            [(2, 3, 5), (13, 12, 11), 0],
            [(0, 0, 0), (1, 1, 1), 0],
        ]
        self.assertListEqual(test, correct)

    def test3_identity_factors(self):
        correct = [
            [(0, 0, 0), (17, 16, 15), 0],
            [(0, 0, 0), (17, 16, 15), 0],
            [(0, 0, 0), (8, 8, 8), 0],
            [(2, 3, 5), (13, 12, 11), 0],
            [(0, 0, 0), (1, 1, 1), 0],
        ]
        for i in range(10):
            test = get_bbox_transform_result(RandomRotate90, get_bbox_test_set(),
                                             params={'factor': i * 4, 'axes': [1, 2, 3]})
            self.assertListEqual(test, correct)

    def test4_iterable_factor(self):
        test = get_bbox_transform_result(RandomRotate90, get_bbox_test_set(), params={'factor': [1], 'axes': [1]})
        correct = [
            [(0, 0, 0), (17, 15, 16), 0],
            [(0, 0, 0), (17, 15, 16), 0],
            [(0, 0, 8), (8, 8, 16), 0],
            [(2, 5, 4), (13, 11, 13), 0],
            [(0, 0, 15), (1, 1, 16), 0],
        ]

        self.assertListEqual(test, correct)

        test = get_bbox_transform_result(RandomRotate90, get_bbox_test_set(),
                                         params={'factor': [2, 2, 2], 'axes': [1, 2, 3]})
        self.assertListEqual(test, get_bbox_test_set())

    def test5_negative_factor(self):
        test = get_bbox_transform_result(RandomRotate90, get_bbox_test_set(), params={'factor': -1, 'axes': [1]})
        self.assertListEqual(test, get_bbox_test_set())

    def test6_single_axis_large_factor(self):
        test = get_bbox_transform_result(RandomRotate90, get_bbox_test_set(), params={'factor': 5, 'axes': [1]})
        correct = [
            [(0, 0, 0), (17, 15, 16), 0],
            [(0, 0, 0), (17, 15, 16), 0],
            [(0, 0, 8), (8, 8, 16), 0],
            [(2, 5, 4), (13, 11, 13), 0],
            [(0, 0, 15), (1, 1, 16), 0],
        ]

        self.assertListEqual(test, correct)


class TestRescaleBBox(unittest.TestCase):
    def test1_upscale_all_integer(self):
        test = get_bbox_transform_result(Rescale, get_bbox_test_set(), params={'scales': 2})
        correct = [
            [(0, 0, 0), (34, 32, 30), 0],
            [(0, 0, 0), (34, 32, 30), 0],
            [(0, 0, 0), (16, 16, 16), 0],
            [(4, 6, 10), (26, 24, 22), 0],
            [(0, 0, 0), (2, 2, 2), 0],
        ]

        self.assertListEqual(test, correct)

    def test2_upscale_or_identity(self):
        test = get_bbox_transform_result(Rescale, get_bbox_test_set(), params={'scales': (1, 2, 1)})
        correct = [
            [(0, 0, 0), (17, 32, 15), 0],
            [(0, 0, 0), (17, 32, 15), 0],
            [(0, 0, 0), (8, 16, 8), 0],
            [(2, 6, 5), (13, 24, 11), 0],
            [(0, 0, 0), (1, 2, 1), 0],
        ]

        self.assertListEqual(test, correct)

    def test3_downscale_upscale_fractional(self):
        test = get_bbox_transform_result(Rescale, get_bbox_test_set(), params={'scales': (.8, .5, 1.4)})
        correct = [
            [(0, 0, 0), (13.6, 8, 21), 0],
            [(0, 0, 0), (13.6, 8, 21), 0],
            [(0, 0, 0), (6.4, 4, 11.2), 0],
            [(1.6, 1.5, 7), (10.4, 6, 15.4), 0],
            [(0, 0, 0), (.8, .5, 1.4), 0],
        ]

        self.assertTrue(are_lists_close(test, correct))

    def test4_squish_bbox(self):
        bbox_list = [((2, 2, 2), (4, 4, 4), 0)]
        test = get_bbox_transform_result(Rescale, bbox_list, params={'scales': .2}, shape=(10, 10, 10))
        correct = [[(0.4, 0.4, 0.4), (0.8, 0.8, 0.8), 0]]
        self.assertTrue(are_lists_close(test, correct))

        bbox_list = [((2, 2, 2), (4, 4, 4), 0)]
        test = get_bbox_transform_result(Rescale, bbox_list, params={'scales': .1}, shape=(10, 10, 10))
        correct = [[(0.2, 0.2, 0.2), (0.4, 0.4, 0.4), 0]]
        self.assertTrue(are_lists_close(test, correct))

    def test5_squish_image(self):
        bbox_list = [[(2, 2, 2), (4, 4, 4), 0]]
        test = get_bbox_transform_result(Rescale, bbox_list, params={'scales': .01}, shape=(10, 10, 10))
        correct = [[(0.02, 0.02, 0.02), (0.04, 0.04, 0.04), 0]]
        self.assertTrue(are_lists_close(test, correct))

        test = get_bbox_transform_result(Rescale, bbox_list, params={'scales': -2}, shape=(10, 10, 10))
        self.assertTrue(are_lists_close(test, bbox_list))


class TestResizeBBox(unittest.TestCase):
    def test1_upscale_all_integer(self):
        test = get_bbox_transform_result(Resize, get_bbox_test_set(), params={'shape': (36, 34, 32)})
        correct = [
            [(0, 0, 0), (34, 32, 30), 0],
            [(0, 0, 0), (34, 32, 30), 0],
            [(0, 0, 0), (16, 16, 16), 0],
            [(4, 6, 10), (26, 24, 22), 0],
            [(0, 0, 0), (2, 2, 2), 0],
        ]

        self.assertListEqual(test, correct)

    def test2_upscale_or_identity(self):
        test = get_bbox_transform_result(Resize, get_bbox_test_set(), params={'shape': (18, 34, 16)})
        correct = [
            [(0, 0, 0), (17, 32, 15), 0],
            [(0, 0, 0), (17, 32, 15), 0],
            [(0, 0, 0), (8, 16, 8), 0],
            [(2, 6, 5), (13, 24, 11), 0],
            [(0, 0, 0), (1, 2, 1), 0],
        ]

        self.assertListEqual(test, correct)

    def test3_downscale_fractional(self):
        test = get_bbox_transform_result(Resize, get_bbox_test_set(), params={'shape': (9, 8, 8)})
        correct = [
            [(0, 0, 0), (8.5, 7.5, 7.5), 0],
            [(0, 0, 0), (8.5, 7.5, 7.5), 0],
            [(0, 0, 0), (4, 3.7, 4), 0],
            [(1, 1.4, 2.5), (6.5, 5.6, 5.5), 0],
            [(0, 0, 0), (.5, .4706, 0.5), 0],
        ]

        self.assertTrue(are_lists_close(test, correct, atol=1e-1))

    def test4_squish_bbox(self):
        bbox_list = [((2, 2, 2), (4, 4, 4), 0)]
        test = get_bbox_transform_result(Resize, bbox_list, params={'shape': (2, 2, 2)}, shape=(10, 10, 10))
        correct = [[(0.4, 0.4, 0.4), (0.8, 0.8, 0.8), 0]]
        self.assertTrue(are_lists_close(test, correct))

        bbox_list = [((2, 2, 2), (4, 4, 4), 0)]
        test = get_bbox_transform_result(Resize, bbox_list, params={'shape': (1, 1, 1)}, shape=(10, 10, 10))
        correct = [[(0.2, 0.2, 0.2), (0.4, 0.4, 0.4), 0]]
        self.assertTrue(are_lists_close(test, correct))

    def test5_squish_image(self):
        bbox_list = [[(2, 2, 2), (4, 4, 4), 0]]
        test = get_bbox_transform_result(Resize, bbox_list, params={'shape': (0.1, 0.1, 0.1)}, shape=(10, 10, 10))
        correct = [[(0.02, 0.02, 0.02), (0.04, 0.04, 0.04), 0]]
        self.assertTrue(are_lists_close(test, correct))

        test = get_bbox_transform_result(Resize, bbox_list, params={'shape': (0, 0, 0)}, shape=(10, 10, 10))
        self.assertTrue(are_lists_close(test, bbox_list))

        test = get_bbox_transform_result(Resize, bbox_list, params={'shape': (-2, -2, -2)}, shape=(10, 10, 10))
        self.assertTrue(are_lists_close(test, bbox_list))

        test = get_bbox_transform_result(Resize, bbox_list, params={'shape': (0, 0, 10)}, shape=(10, 10, 10))
        self.assertTrue(are_lists_close(test, bbox_list))

        test = get_bbox_transform_result(Resize, bbox_list, params={'shape': (-2, 10, 10)}, shape=(10, 10, 10))
        self.assertTrue(are_lists_close(test, bbox_list))


class TestPadBBox(unittest.TestCase):
    def test1_pad_all(self):
        test = get_bbox_transform_result(Pad, get_bbox_test_set(), params={'pad_size': 10})
        correct = [
            [(10, 10, 10), (27, 26, 25), 0],
            [(10, 10, 10), (27, 26, 25), 0],
            [(10, 10, 10), (18, 18, 18), 0],
            [(12, 13, 15), (23, 22, 21), 0],
            [(10, 10, 10), (11, 11, 11), 0],
        ]
        self.assertListEqual(test, correct)

    def test2_pad_or_nopad(self):
        test = get_bbox_transform_result(Pad, get_bbox_test_set(), params={'pad_size': (10, 0)})
        correct = [
            [(10, 10, 10), (27, 26, 25), 0],
            [(10, 10, 10), (27, 26, 25), 0],
            [(10, 10, 10), (18, 18, 18), 0],
            [(12, 13, 15), (23, 22, 21), 0],
            [(10, 10, 10), (11, 11, 11), 0],
        ]
        self.assertListEqual(test, correct)

    def test3_pad_or_nopad(self):
        test = get_bbox_transform_result(Pad, get_bbox_test_set(), params={'pad_size': (0, 0, 10, 10, 2, 2)})
        correct = [
            [(0, 10, 2), (17, 26, 17), 0],
            [(0, 10, 2), (17, 26, 17), 0],
            [(0, 10, 2), (8, 18, 10), 0],
            [(2, 13, 7), (13, 22, 13), 0],
            [(0, 10, 2), (1, 11, 3), 0],
        ]
        self.assertListEqual(test, correct)

    def test4_negative_padding(self):
        test = get_bbox_transform_result(Pad, get_bbox_test_set(), params={'pad_size': (-1, 0)})
        self.assertTrue(are_lists_close(test, get_bbox_test_set()))

    # unsupported, unexpected scenarios
    #
    # def test5_bbox_inclusion(self):
    #     # bbox outside of image domain that suddenly appears inside the image domain due to padding
    #     bbox_list = [[(12, 12, 12), (14, 14, 14), 0]]
    #     test = get_bbox_transform_result(Pad, bbox_list, params={'pad_size': 10}, shape=(10, 10, 10))
    #     self.assertListEqual(test, [])
    #
    # def test6_bbox_partial_inclusion(self):
    #     # bbox partially in the image domain (half in, half out) that suddenly appears fully inside the image domain due to padding
    #     bbox_list = [[(8, 8, 8), (14, 14, 14), 0]]
    #     test = get_bbox_transform_result(Pad, bbox_list, params={'pad_size': 10}, shape=(10, 10, 10))
    #     correct = [[(18, 18, 18), (19, 19, 19), 0]]
    #     self.assertTrue(are_lists_close(test, correct))
    #
    # def test7_bbox_outside(self):
    #     # bbox outside of image domain that stays outside it even after padding
    #     bbox_list = [[(12, 12, 12), (14, 14, 14), 0]]
    #     test = get_bbox_transform_result(Pad, bbox_list, params={'pad_size': 1}, shape=(10, 10, 10))
    #     self.assertListEqual(test, [])


class TestCenterCrop(unittest.TestCase):
    def test1_crop_all(self):
        test = get_bbox_transform_result(CenterCrop, get_bbox_test_set(), params={'shape': (10, 10, 10)})
        correct = [
            [(0, 0, 0), (9, 9, 9), 0],
            [(0, 0, 0), (9, 9, 9), 0],
            [(0, 0, 0), (4, 5, 5), 0],
            [(0, 0, 2), (9, 9, 8), 0],
        ]
        self.assertListEqual(test, correct)

    def test2_crop_to_zero(self):
        test = get_bbox_transform_result(CenterCrop, get_bbox_test_set(), params={'shape': (0, 0, 0)})
        self.assertListEqual(test, get_bbox_test_set())

    def test2_crop_to_ones(self):
        test = get_bbox_transform_result(CenterCrop, get_bbox_test_set(), params={'shape': (1, 1, 1)})

        # # if the bbox could cover 1 pixel (the endpoint is exclusive):
        # correct = [
        #     [(0, 0, 0), (1, 1, 1), 0],
        #     [(0, 0, 0), (1, 1, 1), 0],
        #     [(0, 0, 0), (1, 1, 1), 0],
        # ]
        # self.assertListEqual(test, correct)

        # since the bboxes would be [(0, 0, 0), (0, 0, 0), 0], we remove them:
        self.assertListEqual(test, [])

    def test3_crop_all_minpercentage(self):
        test = get_bbox_transform_result(CenterCrop, get_bbox_test_set(),
                                         params={'shape': (10, 10, 10), 'min_percentage': 0.5})
        correct = [
            [(0, 0, 2), (9, 9, 8), 0]
        ]
        self.assertListEqual(test, correct)

    def test4_crop_all_minvolume(self):
        test = get_bbox_transform_result(CenterCrop, get_bbox_test_set(),
                                         params={'shape': (10, 10, 10), 'min_volume': 400})
        correct = [
            [(0, 0, 0), (9, 9, 9), 0],
            [(0, 0, 0), (9, 9, 9), 0],
            [(0, 0, 2), (9, 9, 8), 0]
        ]
        self.assertListEqual(test, correct)

    def test5_crop_to_one_keepall(self):
        test = get_bbox_transform_result(CenterCrop, get_bbox_test_set(), params={'shape': (1, 1, 1), 'keep_all': True})
        correct = [
            [(-8, -8, -7), (9, 8, 8), 0],
            [(-8, -8, -7), (9, 8, 8), 0],
            [(-8, -8, -7), (0, 0, 1), 0],
            [(-6, -5, -2), (5, 4, 4), 0],
            [(-8, -8, -7), (-7, -7, -6), 0],
        ]
        self.assertListEqual(test, correct)

    def test6_odd_cropping(self):
        bbox_list = [((2, 2, 2), (4, 4, 4), 0)]
        img = np.arange(10**3).reshape((1, 10, 10, 10))
        pipeline = Compose([CenterCrop(shape=(9, 9, 9))])
        res_dict = pipeline(image=img, bboxes=bbox_list)

        img_res = res_dict['image']
        self.assertEqual(img_res[0, 0, 0, 0], 0)

        test = res_dict['bboxes']
        self.assertTrue(are_lists_close(test, bbox_list))

    def test7_exclude_bbox(self):
        bbox_list = [[(0, 0, 0), (2, 2, 2), 0]]

        # bbox fully in -> partially in
        test = get_bbox_transform_result(CenterCrop, bbox_list, params={'shape': (8, 8, 8), 'min_volume': 1}, shape=(10, 10, 10))
        correct = [[(0, 0, 0), (1, 1, 1), 0]]
        self.assertTrue(are_lists_close(test, correct))

        # bbox fully in -> fully out
        test = get_bbox_transform_result(CenterCrop, bbox_list, params={'shape': (2, 2, 2), 'min_volume': 1}, shape=(10, 10, 10))
        self.assertListEqual(test, [])

        # bbox fully in -> fully out in one axis
        test = get_bbox_transform_result(CenterCrop, bbox_list, params={'shape': (10, 2, 10), 'min_volume': 1}, shape=(10, 10, 10))
        self.assertListEqual(test, [])


class TestScaleBBox(unittest.TestCase):
    def test1_upscale_all_integer(self):
        test = get_bbox_transform_result(Scale, get_bbox_test_set(), params={'scales': 2})
        correct = [
            [(0, 0, 0), (17, 16, 15), 0],
            [(0, 0, 0), (17, 16, 15), 0],
            [(0, 0, 0), (7.5, 8, 8.5), 0],
            [(0, 0, 2.5), (17, 16, 14.5), 0],
        ]

        self.assertListEqual(test, correct)

    def test2_upscale_or_identity(self):
        test = get_bbox_transform_result(Scale, get_bbox_test_set(), params={'scales': (1, 2, 1)})
        correct = [
            [(0, 0, 0), (17, 16, 15), 0],
            [(0, 0, 0), (17, 16, 15), 0],
            [(0, 0, 0), (8, 8, 8), 0],
            [(2, 0, 5), (13, 16, 11), 0],
        ]

        self.assertListEqual(test, correct)

    # def test3_downscale_upscale_fractional(self):
    #     test = get_bbox_transform_result(Scale, get_bbox_test_set(), params={'scales': (.8, .5, 1.4)})
    #     correct = [
    #         [(0, 0, 0), (13.6, 8, 21), 0],
    #         [(0, 0, 0), (13.6, 8, 21), 0],
    #         [(0, 0, 0), (6.4, 4, 11.2), 0],
    #         [(1.6, 1.5, 7), (10.4, 6, 15.4), 0],
    #     ]
    #
    #     self.assertTrue(are_lists_close(test, correct))

    def test3_squish_bbox(self):
        bbox_list = [((0, 0, 0), (9, 9, 9), 0)]
        img = np.arange(10**3).reshape((1, 10, 10, 10))
        pipeline = Compose([Scale(scales=.2)])
        res_dict = pipeline(image=img, bboxes=bbox_list)

        img_res = res_dict['image']
        self.assertTupleEqual(img_res.shape, (1, 10, 10, 10))

        test = res_dict['bboxes']
        correct = [[(3.6, 3.6, 3.6), (5.4, 5.4, 5.4), 0]]
        self.assertTrue(are_lists_close(test, correct))

        bbox_list = [((0, 0, 0), (9, 9, 9), 0)]
        test = get_bbox_transform_result(Scale, bbox_list, params={'scales': .1}, shape=(10, 10, 10))
        correct = [[(4.05, 4.05, 4.05), (4.95, 4.95, 4.95), 0]]  # --> empty
        self.assertTrue(are_lists_close(test, correct))

    def test4_squish_bbox(self):
        bbox_list = [((2, 2, 2), (4, 4, 4), 0)]
        img = np.arange(10**3).reshape((1, 10, 10, 10))
        pipeline = Compose([Scale(scales=.2)])
        res_dict = pipeline(image=img, bboxes=bbox_list)

        img_res = res_dict['image']
        self.assertTupleEqual(img_res.shape, (1, 10, 10, 10))

        test = res_dict['bboxes']
        correct = [[(4, 4, 4), (4.4, 4.4, 4.4), 0]]
        self.assertTrue(are_lists_close(test, correct))

        bbox_list = [((2, 2, 2), (4, 4, 4), 0)]
        test = get_bbox_transform_result(Scale, bbox_list, params={'scales': .1}, shape=(10, 10, 10))
        correct = [[(4.25, 4.25, 4.25), (4.45, 4.45, 4.45), 0]]
        self.assertTrue(are_lists_close(test, correct))

    def test5_squish_image(self):
        bbox_list = [[(2, 2, 2), (4, 4, 4), 0]]
        test = get_bbox_transform_result(Scale, bbox_list, params={'scales': .01}, shape=(10, 10, 10))
        correct = [[(4.475, 4.475, 4.475), (4.495, 4.495, 4.495), 0]]
        self.assertTrue(are_lists_close(test, correct))

        test = get_bbox_transform_result(Scale, bbox_list, params={'scales': 0}, shape=(10, 10, 10))
        self.assertTrue(are_lists_close(test, bbox_list))

        test = get_bbox_transform_result(Scale, bbox_list, params={'scales': -2}, shape=(10, 10, 10))
        self.assertTrue(are_lists_close(test, bbox_list))


class TestAffine(unittest.TestCase):
    def test_identity(self):
        test = get_bbox_transform_result(AffineTransform, get_bbox_test_set(), params={})  # identity
        self.assertListEqual(test, get_bbox_test_set())

    def test_translation_positive(self):
        test = get_bbox_transform_result(AffineTransform, get_bbox_test_set(), params={'translation': (1, 1, 1)})
        correct = [
            [(1, 1, 1), (17, 16, 15), 0],
            [(1, 1, 1), (17, 16, 15), 0],
            [(1, 1, 1), (9, 9, 9), 0],
            [(3, 4, 6), (14, 13, 12), 0],
            [(1, 1, 1), (2, 2, 2), 0],
        ]
        self.assertTrue(are_lists_close(test, correct))

    def test_translation_positive_fractional(self):
        test = get_bbox_transform_result(AffineTransform, get_bbox_test_set(), params={'translation': (0.5, 0.5, 0.5)})
        correct = [
            [(0.5, 0.5, 0.5), (17, 16, 15), 0],
            [(0.5, 0.5, 0.5), (17, 16, 15), 0],
            [(0.5, 0.5, 0.5), (8.5, 8.5, 8.5), 0],
            [(2.5, 3.5, 5.5), (13.5, 12.5, 11.5), 0],
            [(0.5, 0.5, 0.5), (1.5, 1.5, 1.5), 0],
        ]
        self.assertTrue(are_lists_close(test, correct))

    def test_translation_negative(self):
        test = get_bbox_transform_result(AffineTransform, get_bbox_test_set(), params={'translation': (-1, -1, -1)})
        correct = [
            [(0, 0, 0), (16, 15, 14), 0],
            [(0, 0, 0), (16, 15, 14), 0],
            [(0, 0, 0), (7, 7, 7), 0],
            [(1, 2, 4), (12, 11, 10), 0],
        ]
        self.assertTrue(are_lists_close(test, correct))

    def test_translation_negative_large(self):
        test = get_bbox_transform_result(AffineTransform, get_bbox_test_set(),
                                         params={'translation': (-100, -100, -100)})
        self.assertListEqual(test, [])

    def test_scale_upscale_all_integer(self):
        test = get_bbox_transform_result(Scale, get_bbox_test_set(), params={'scales': 2})
        correct = [
            [(0, 0, 0), (17, 16, 15), 0],
            [(0, 0, 0), (17, 16, 15), 0],
            [(0, 0, 0), (7.5, 8, 8.5), 0],
            [(0, 0, 2.5), (17, 16, 14.5), 0],
        ]

        self.assertListEqual(test, correct)

    def test_scale_upscale_or_identity(self):
        test = get_bbox_transform_result(Scale, get_bbox_test_set(), params={'scales': (1, 2, 1)})
        correct = [
            [(0, 0, 0), (17, 16, 15), 0],
            [(0, 0, 0), (17, 16, 15), 0],
            [(0, 0, 0), (8, 8, 8), 0],
            [(2, 0, 5), (13, 16, 11), 0],
        ]

        self.assertListEqual(test, correct)

    def test_scale_squish_bbox(self):
        bbox_list = [((0, 0, 0), (9, 9, 9), 0)]
        img = np.arange(10**3).reshape((1, 10, 10, 10))
        pipeline = Compose([Scale(scales=.2)])
        res_dict = pipeline(image=img, bboxes=bbox_list)

        img_res = res_dict['image']
        self.assertTupleEqual(img_res.shape, (1, 10, 10, 10))

        test = res_dict['bboxes']
        correct = [[(3.6, 3.6, 3.6), (5.4, 5.4, 5.4), 0]]
        self.assertTrue(are_lists_close(test, correct))

        bbox_list = [((0, 0, 0), (9, 9, 9), 0)]
        test = get_bbox_transform_result(Scale, bbox_list, params={'scales': .1}, shape=(10, 10, 10))
        correct = [[(4.05, 4.05, 4.05), (4.95, 4.95, 4.95), 0]]  # --> empty
        self.assertTrue(are_lists_close(test, correct))

    def test_scale_squish_bbox_2(self):
        bbox_list = [((2, 2, 2), (4, 4, 4), 0)]
        img = np.arange(10**3).reshape((1, 10, 10, 10))
        pipeline = Compose([Scale(scales=.2)])
        res_dict = pipeline(image=img, bboxes=bbox_list)

        img_res = res_dict['image']
        self.assertTupleEqual(img_res.shape, (1, 10, 10, 10))

        test = res_dict['bboxes']
        correct = [[(4, 4, 4), (4.4, 4.4, 4.4), 0]]
        self.assertTrue(are_lists_close(test, correct))

        bbox_list = [((2, 2, 2), (4, 4, 4), 0)]
        test = get_bbox_transform_result(Scale, bbox_list, params={'scales': .1}, shape=(10, 10, 10))
        correct = [[(4.25, 4.25, 4.25), (4.45, 4.45, 4.45), 0]]
        self.assertTrue(are_lists_close(test, correct))

    def test_scale_squish_image(self):
        bbox_list = [[(2, 2, 2), (4, 4, 4), 0]]
        test = get_bbox_transform_result(Scale, bbox_list, params={'scales': .01}, shape=(10, 10, 10))
        correct = [[(4.475, 4.475, 4.475), (4.495, 4.495, 4.495), 0]]
        self.assertTrue(are_lists_close(test, correct))

        test = get_bbox_transform_result(Scale, bbox_list, params={'scales': 0}, shape=(10, 10, 10))
        self.assertTrue(are_lists_close(test, bbox_list))

        test = get_bbox_transform_result(Scale, bbox_list, params={'scales': -2}, shape=(10, 10, 10))
        self.assertTrue(are_lists_close(test, bbox_list))

    def test_rotation_360(self):
        test = get_bbox_transform_result(AffineTransform, get_bbox_test_set(), params={'angles': (360, 360, 360)})
        self.assertTrue(are_lists_close(test, get_bbox_test_set()))

    def test_rotation_symmetry(self):
        test = get_bbox_transform_result(AffineTransform, [[(0, 0, 0), (1, 2, 3), 0]], params={'angles': (90, 0, 10)})
        test2 = get_bbox_transform_result(AffineTransform, [[(0, 0, 0), (1, 2, 3), 0]],
                                          params={'angles': (-270, 0, -350)})

        self.assertTrue(are_lists_close(test, test2))

    def test_rotation_small_bbox_45(self):
        # the library uses rotation around image center:
        test = get_bbox_transform_result(AffineTransform, [[(3, 3, 3), (5, 5, 5), 0]], shape=(9, 9, 9),
                                         params={'angles': (45, 0, 0)})
        correct = [
            [(3, 4-np.sqrt(2), 4-np.sqrt(2)), (5, 4+np.sqrt(2), 4+np.sqrt(2)), 0]
        ]
        self.assertTrue(are_lists_close(test, correct, atol=1e-1))

    def test_rotation_small_bbox_45_minvolume(self):
        # rotation by 45 degrees for a bbox too small for the min_volume threshold -> big enough for the threshold

        test = get_bbox_transform_result(AffineTransform, [[(3, 3, 3), (5, 5, 5), 0]], shape=(9, 9, 9),
                                         params={'angles': (0, 0, 0), 'min_volume': 10})
        self.assertListEqual(test, [])

        test = get_bbox_transform_result(AffineTransform, [[(3, 3, 3), (5, 5, 5), 0]], shape=(9, 9, 9),
                                         params={'angles': (45, 0, 0), 'min_volume': 10})
        correct = [
            [(3, 4-np.sqrt(2), 4-np.sqrt(2)), (5, 4+np.sqrt(2), 4+np.sqrt(2)), 0]
        ]
        self.assertTrue(are_lists_close(test, correct, atol=1e-1))

    def test_rotation_any_degree(self):
        test = get_bbox_transform_result(AffineTransform, get_bbox_test_set(), params={'angles': (60, 27, 118)})
        print(test)


#########################################################################
#                                                                       #
#                      Format conversion tests                          #
#                                                                       #
#########################################################################


def get_bbox_convert_result(formatt,
                            data: list[list[int]],
                            shape: tuple[int, int, int] = (9, 9, 9),
                            params: dict = {}):
    w, h, d = shape
    sample = {
        'image': np.zeros((4, w, h, d)),
        'bboxes': data
    }

    transformation = Compose([AffineTransform(p=1)], bbox_format=formatt)
    transformation_res = transformation(**sample)

    return transformation_res['bboxes']


class TestBBoxFormatConversion(unittest.TestCase):
    def testvoc(self):
        bbox = [[(0, 1, 2), (3, 4, 5), 0]]
        test = get_bbox_convert_result('voc', bbox)

        self.assertListEqual(bbox, test)

    def testcoco(self):
        bbox = [[(0, 1, 2), (3, 4, 5), 0]]
        test = get_bbox_convert_result('coco', bbox)

        self.assertListEqual(bbox, test)

    def testalbm(self):
        bbox = [[(0.1, 0.2, 0.3), (0.3, 0.4, 0.5), 0]]
        test = get_bbox_convert_result('albumentations', bbox)

        self.assertTrue(are_lists_close(bbox, test))

    def testyolo(self):
        bbox = [[(0.1, 0.2, 0.3), (0.1, 0.1, 0.2), 0]]
        test = get_bbox_convert_result('yolo', bbox)

        self.assertTrue(are_lists_close(bbox, test))


if __name__ == '__main__':
    unittest.main()
