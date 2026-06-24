# ============================================================================================= #
#  Author:       Filip Lux, Lucia Hradecká                                                      #
#  Copyright:    Filip Lux          : lux.filip@gmail.com                                       #
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

from src.bio_volumentations.augmentations.transforms import (
    GaussianNoise, PoissonNoise, Resize, Pad, Scale, Flip, CenterCrop, AffineTransform,
    RandomScale, RandomRotate90, RandomFlip, RandomCrop, RandomAffineTransform, RandomGamma,
    NormalizeMeanStd, GaussianBlur, Normalize, HistogramEqualization, RandomBrightnessContrast,
    RandomGaussianBlur, RemoveBackgroundGaussian, Rescale)
from src.bio_volumentations.core.composition import Compose

DEBUG = False


def get_keypoints_tests(transform,
                        in_shape: tuple = (32, 33, 34),
                        params: dict = {}):
    w, h, d = in_shape

    img = np.zeros((4, w, h, d), np.float32)
    mask = np.zeros((w, h, d), np.int32)
    keypoints = []

    lbd = 3

    for _ in range(15):
        w1, h1, d1 = np.random.randint(lbd, w - lbd), \
            np.random.randint(lbd, h - lbd), \
            np.random.randint(lbd, d - lbd)
        img[:, w1 - lbd:w1 + lbd, h1 - lbd:h1 + lbd, d1 - lbd:d1 + lbd] = 10.
        mask[w1 - lbd:w1 + lbd, h1 - lbd:h1 + lbd, d1 - lbd:d1 + lbd] = 10.
        keypoints.append((w1 - 0., h1 - 0., d1 - 0.))

    sample = {'image': img,
              'mask': mask,
              'keypoints': keypoints}

    tr_indiv = transform(**params, p=1)
    tr = Compose([tr_indiv])
    sample_transformed = tr(**sample)

    keypoints_transformed = sample_transformed['keypoints']
    if DEBUG:
        print('KEYPOINTS', transform, keypoints)
        print('KEYPOINTS TRANSFORMED', transform, keypoints_transformed)

    tests = []
    for k in keypoints_transformed:
        coos = (np.array(k) + .5).astype(int)
        tests.append((sample_transformed['image'][0, coos[0], coos[1], coos[2]], 10.,
                      f'img, {k} {coos} {tr_indiv}'))
        tests.append((sample_transformed['mask'][coos[0], coos[1], coos[2]], 10.,
                      f'mask {k} {coos} {tr_indiv}'))

    return tests


def get_shape_tests(transform,
                    in_shape: tuple,
                    params={},
                    exp_shape=None):
    """
    Iterates over all the possibilities, hot the array can passed throught the transform
    Args:
        transform: biovol transform,
        in_shape: spatial dimension of the input image
        params: optional, params of the biovol transform

    Returns:
        list of outputs and expected shapes

    """

    w, h, d = in_shape
    if exp_shape is not None:
        w_, h_, d_ = exp_shape
    else:
        w_, h_, d_ = params['shape'] if 'shape' in params.keys() else (w, h, d)

    res = []
    tr = Compose([transform(**params, p=1)])

    # img (W, H, D), mask (W, H, D)
    img = np.ones((w, h, d), dtype=np.float32)
    mask = np.ones((w, h, d), dtype=np.int32)
    fmask = np.ones((w, h, d), dtype=np.float32)
    # print(img.dtype, mask.dtype, fmask.dtype)
    tr_img = tr(image=img, mask=mask, float_mask=fmask)
    # print(tr_img['image'].dtype, tr_img['mask'].dtype, tr_img['float_mask'].dtype)
    res.append((tr_img['image'], (1, w_, h_, d_), np.float32))
    res.append((tr_img['mask'], (w_, h_, d_), np.int32))
    res.append((tr_img['float_mask'], (w_, h_, d_), np.float32))

    # img (C, W, H, D), mask (W, H, D)
    img = np.ones((4, w, h, d), dtype=np.single)
    mask = np.ones((w, h, d), dtype=int)
    fmask = np.ones((w, h, d), dtype=np.single)
    tr_img = tr(image=img, mask=mask, float_mask=fmask)
    res.append((tr_img['image'], (4, w_, h_, d_), np.float32))
    res.append((tr_img['mask'], (w_, h_, d_), np.int32))
    res.append((tr_img['float_mask'], (w_, h_, d_), np.float32))

    # img (C, W, H, D, T), mask (W, H, D, T)
    img = np.ones((4, w, h, d, 5), dtype=np.single)
    mask = np.ones((w, h, d, 5), dtype=int)
    fmask = np.ones((w, h, d, 5), dtype=np.single)
    tr_img = tr(image=img, mask=mask, float_mask=fmask)
    res.append((tr_img['image'], (4, w_, h_, d_, 5), np.float32))
    res.append((tr_img['mask'], (w_, h_, d_, 5), np.int32))
    res.append((tr_img['float_mask'], (w_, h_, d_, 5), np.float32))

    return res


def get_shape_tests_5d(transform, in_shape: tuple, params={}):
    """
    Iterates over all the possibilities, hot the array can passed throught the transform
    Args:
        transform: biovol transform,
        in_shape: spatial dimension of the input image
        params: optional, params of the biovol transform

    Returns:
        list of outputs and expected shapes

    """

    c, w, h, d, t = in_shape
    w_, h_, d_ = params['shape'] if 'shape' in params.keys() else (w, h, d)

    res = []
    tr = Compose([transform(**params, p=1)])

    # img (C, W, H, D, T), mask (W, H, D, T)
    img = np.ones((c, w, h, d, t), dtype=np.single)
    mask = np.ones((w, h, d, t), dtype=int)
    fmask = np.ones((w, h, d, t), dtype=np.single)
    tr_img = tr(image=img, mask=mask, float_mask=fmask)
    res.append((tr_img['image'], (c, w_, h_, d_, t), np.float32))
    res.append((tr_img['mask'], (w_, h_, d_, t), np.int32))
    res.append((tr_img['float_mask'], (w_, h_, d_, t), np.float32))

    return res


class TestScale(unittest.TestCase):
    def test_shape(self):
        tests = get_shape_tests(Scale, (31, 32, 33), params={'scales': 1.5})
        for tr_img, expected_shape, data_type in tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)

        tests = get_shape_tests(Scale, (31, 32, 33), params={'scales': 0.8})
        for tr_img, expected_shape, data_type in tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)

    def test_keypoints(self):

        tests = get_keypoints_tests(Scale, params={'scales': 1.5})
        for value, expected_value, msg in tests:
            self.assertGreater(value, expected_value * 0.1, msg)

        tests = get_keypoints_tests(Scale, params={'scales': 0.8})
        for value, expected_value, msg in tests:
            self.assertGreater(value, expected_value * 0.1, msg)


class TestRandomScale(unittest.TestCase):
    def test_shape(self):

        limits = [0.2,
                  (0.8, 1.2),
                  (0.2, 0.3, 0.1),
                  (0.8, 1.2, 0.9, 1.1, 0.7, 1.)]

        for scaling_limit in limits:
            tests = get_shape_tests(RandomScale,
                                    in_shape=(31, 32, 33),
                                    params={'scaling_limit': scaling_limit})
            for tr_img, expected_shape, data_type in tests:
                self.assertTupleEqual(tr_img.shape, expected_shape)
                self.assertEqual(tr_img.dtype, data_type)

    def test_keypoints(self):

        limits = [0.2,
                  (0.8, 1.2),
                  (0.2, 0.3, 0.1),
                  (0.8, 1.2, 0.9, 1.1, 0.7, 1.)]

        for scaling_limit in limits:
            tests = get_keypoints_tests(RandomScale,
                                        in_shape=(61, 62, 63),
                                        params={'scaling_limit': scaling_limit})
            for value, expected_value, msg in tests:
                self.assertGreater(value, expected_value * 0.5, msg)


class TestRandomRotate90(unittest.TestCase):
    def test_shape(self):

        axes_list = [[1], [2], [3],
                     [1, 2],
                     [1, 2, 3], None,
                     [1, 2, 3, 2, 3, 1, 3]]

        for axes in axes_list:
            tests = get_shape_tests(RandomRotate90, (30, 30, 30),
                                    params={'axes': axes})
            for tr_img, expected_shape, data_type in tests:
                self.assertTupleEqual(tr_img.shape, expected_shape)
                self.assertEqual(tr_img.dtype, data_type)

    def test_keypoints(self):

        axes_list = [[1], [2], [3],
                     [1, 2],
                     [1, 2, 3], None,
                     [1, 2, 3, 2, 3, 1, 3]]

        for _ in range(32):
            for axes in axes_list:
                tests = get_keypoints_tests(RandomRotate90, params={'axes': axes})
                for value, expected_value, msg in tests:
                    self.assertGreater(value, expected_value * 0.1, msg)

    def perform_tr(self, tr, kpts_in, img_shape_in, kpts_exp, img_shape_exp=None, print_out=False):
        pipeline = Compose([tr])

        img = np.zeros(img_shape_in, dtype=float)

        res_dict = pipeline(image=img, keypoints=kpts_in)

        if img_shape_exp is not None:
            if len(img_shape_in) > 3:
                self.assertTupleEqual(res_dict['image'].shape, img_shape_exp)
            else:
                self.assertTupleEqual(res_dict['image'].shape[1:], img_shape_exp)

        kpts_out = res_dict['keypoints']

        if print_out:
            print(f'exp: {kpts_exp}')
            print(f'out: {kpts_out}')

        for idx in range(len(kpts_exp)):
            self.assertTupleEqual(kpts_out[idx], kpts_exp[idx])

    def test_keypoints_2(self):
        transform = RandomRotate90(axes=[1], shuffle_axis=False, factor=1, always_apply=True)
        kpts_in = [(0, 0, 0), (1, 2, 3), (4, 5, 6), (3, 2, 1), (9, 9, 9)]
        kpts_exp = [(0, 0, 9), (1, 3, 7), (4, 6, 4), (3, 1, 7), (9, 9, 0)]
        self.perform_tr(transform, kpts_in, (10, 10, 10), kpts_exp, print_out=False)

    def test_keypoints_3(self):
        transform = RandomRotate90(axes=[1], shuffle_axis=False, factor=1, always_apply=True)
        kpts_in = [(0, 0, 0), (0, 0, 9), (0, 8, 0), (0, 8, 9)]
        kpts_exp = [(0, 0, 8), (0, 9, 8), (0, 0, 0), (0, 9, 0)]
        self.perform_tr(transform, kpts_in, (8, 9, 10), kpts_exp, print_out=False)

        kpts_in = [(0, 2, 3), (4, 5, 6), (6, 2, 1)]
        kpts_exp = [(0, 3, 6), (4, 6, 3), (6, 1, 6)]
        self.perform_tr(transform, kpts_in, (8, 9, 10), kpts_exp, print_out=False)

    def test_keypoints_shape_corners(self):
        # around z axis
        transform = RandomRotate90(axes=[1], shuffle_axis=False, factor=1, always_apply=True)
        kpts_in = [(0, 0, 0), (0, 0, 9), (0, 5, 9), (0, 5, 0), (2, 0, 0), (2, 0, 9), (2, 5, 9), (2, 5, 0)]
        kpts_exp = [(0, 0, 5), (0, 9, 5), (0, 9, 0), (0, 0, 0), (2, 0, 5), (2, 9, 5), (2, 9, 0), (2, 0, 0)]
        self.perform_tr(transform, kpts_in, (3, 6, 10), kpts_exp, img_shape_exp=(3, 10, 6), print_out=False)

        transform = RandomRotate90(axes=[1], shuffle_axis=False, factor=2, always_apply=True)
        kpts_in = [(0, 0, 0), (0, 0, 9), (0, 5, 9), (0, 5, 0), (2, 0, 0), (2, 0, 9), (2, 5, 9), (2, 5, 0)]
        kpts_exp = [(0, 5, 9), (0, 5, 0), (0, 0, 0), (0, 0, 9), (2, 5, 9), (2, 5, 0), (2, 0, 0), (2, 0, 9)]
        self.perform_tr(transform, kpts_in, (3, 6, 10), kpts_exp, img_shape_exp=(3, 6, 10), print_out=False)

        transform = RandomRotate90(axes=[1], shuffle_axis=False, factor=3, always_apply=True)
        kpts_in = [(0, 0, 0), (0, 0, 9), (0, 5, 9), (0, 5, 0), (2, 0, 0), (2, 0, 9), (2, 5, 9), (2, 5, 0)]
        kpts_exp = [(0, 9, 0), (0, 0, 0), (0, 0, 5), (0, 9, 5), (2, 9, 0), (2, 0, 0), (2, 0, 5), (2, 9, 5)]
        self.perform_tr(transform, kpts_in, (3, 6, 10), kpts_exp, img_shape_exp=(3, 10, 6), print_out=False)

        # around y axis
        transform = RandomRotate90(axes=[2], shuffle_axis=False, factor=1, always_apply=True)
        kpts_in = [(0, 0, 0), (0, 0, 9), (0, 5, 9), (0, 5, 0), (2, 0, 0), (2, 0, 9), (2, 5, 9), (2, 5, 0)]
        kpts_exp = [(9, 0, 0), (0, 0, 0), (0, 5, 0), (9, 5, 0), (9, 0, 2), (0, 0, 2), (0, 5, 2), (9, 5, 2)]
        self.perform_tr(transform, kpts_in, (3, 6, 10), kpts_exp, img_shape_exp=(10, 6, 3), print_out=False)

        transform = RandomRotate90(axes=[2], shuffle_axis=False, factor=2, always_apply=True)
        kpts_in = [(0, 0, 0), (0, 0, 9), (0, 5, 9), (0, 5, 0), (2, 0, 0), (2, 0, 9), (2, 5, 9), (2, 5, 0)]
        kpts_exp = [(2, 0, 9), (2, 0, 0), (2, 5, 0), (2, 5, 9), (0, 0, 9), (0, 0, 0), (0, 5, 0), (0, 5, 9)]
        self.perform_tr(transform, kpts_in, (3, 6, 10), kpts_exp, img_shape_exp=(3, 6, 10), print_out=False)

        transform = RandomRotate90(axes=[2], shuffle_axis=False, factor=3, always_apply=True)
        kpts_in = [(0, 0, 0), (0, 0, 9), (0, 5, 9), (0, 5, 0), (2, 0, 0), (2, 0, 9), (2, 5, 9), (2, 5, 0)]
        kpts_exp = [(0, 0, 2), (9, 0, 2), (9, 5, 2), (0, 5, 2), (0, 0, 0), (9, 0, 0), (9, 5, 0), (0, 5, 0)]
        self.perform_tr(transform, kpts_in, (3, 6, 10), kpts_exp, img_shape_exp=(10, 6, 3), print_out=False)

        # around x axis
        transform = RandomRotate90(axes=[3], shuffle_axis=False, factor=1, always_apply=True)
        kpts_in = [(0, 0, 0), (0, 0, 9), (0, 5, 9), (0, 5, 0), (2, 0, 0), (2, 0, 9), (2, 5, 9), (2, 5, 0)]
        kpts_exp = [(0, 2, 0), (0, 2, 9), (5, 2, 9), (5, 2, 0), (0, 0, 0), (0, 0, 9), (5, 0, 9), (5, 0, 0)]
        self.perform_tr(transform, kpts_in, (3, 6, 10), kpts_exp, img_shape_exp=(6, 3, 10), print_out=False)

        transform = RandomRotate90(axes=[3], shuffle_axis=False, factor=2, always_apply=True)
        kpts_in = [(0, 0, 0), (0, 0, 9), (0, 5, 9), (0, 5, 0), (2, 0, 0), (2, 0, 9), (2, 5, 9), (2, 5, 0)]
        kpts_exp = [(2, 5, 0), (2, 5, 9), (2, 0, 9), (2, 0, 0), (0, 5, 0), (0, 5, 9), (0, 0, 9), (0, 0, 0)]
        self.perform_tr(transform, kpts_in, (3, 6, 10), kpts_exp, img_shape_exp=(3, 6, 10), print_out=False)

        transform = RandomRotate90(axes=[3], shuffle_axis=False, factor=3, always_apply=True)
        kpts_in = [(0, 0, 0), (0, 0, 9), (0, 5, 9), (0, 5, 0), (2, 0, 0), (2, 0, 9), (2, 5, 9), (2, 5, 0)]
        kpts_exp = [(5, 0, 0), (5, 0, 9), (0, 0, 9), (0, 0, 0), (5, 2, 0), (5, 2, 9), (0, 2, 9), (0, 2, 0)]
        self.perform_tr(transform, kpts_in, (3, 6, 10), kpts_exp, img_shape_exp=(6, 3, 10), print_out=False)

    def test_keypoints_shape_factor0(self):
        for axes in [[1], [2], [3], None, [1, 2, 3], [1, 2, 3, 1, 2, 3, 3], [1, 3], [3, 2]]:
            transform = RandomRotate90(axes=axes, shuffle_axis=False, factor=0, always_apply=True)
            kpts_in = [(0, 0, 0), (0, 0, 9), (0, 5, 9), (0, 5, 0), (2, 0, 0), (2, 0, 9), (2, 5, 9), (2, 5, 0)]
            kpts_exp = [(0, 0, 0), (0, 0, 9), (0, 5, 9), (0, 5, 0), (2, 0, 0), (2, 0, 9), (2, 5, 9), (2, 5, 0)]
            self.perform_tr(transform, kpts_in, (3, 6, 10), kpts_exp, img_shape_exp=(3, 6, 10), print_out=False)

    def test_keypoints_shape_factor2(self):
        for axes in [[1], [2], [3]]:
            transform2 = RandomRotate90(axes=axes, shuffle_axis=False, factor=2, always_apply=True)
            transform1 = RandomRotate90(axes=axes, shuffle_axis=False, factor=1, always_apply=True)
            kpts_in = [(0, 0, 0), (0, 0, 9), (0, 5, 9), (0, 5, 0), (2, 0, 0), (2, 0, 9), (2, 5, 9), (2, 5, 0)]

            pip1 = Compose([transform1, transform1])
            kpts_exp = pip1(image=np.zeros((3, 6, 10)), keypoints=kpts_in)['keypoints']
            self.perform_tr(transform2, kpts_in, (3, 6, 10), kpts_exp, img_shape_exp=(3, 6, 10), print_out=False)

    def test_keypoints_shape_factor3(self):
        for axes in [[1], [2], [3]]:
            transform3 = RandomRotate90(axes=axes, shuffle_axis=False, factor=3, always_apply=True)
            transform1 = RandomRotate90(axes=axes, shuffle_axis=False, factor=1, always_apply=True)
            kpts_in = [(0, 0, 0), (0, 0, 9), (0, 5, 9), (0, 5, 0), (2, 0, 0), (2, 0, 9), (2, 5, 9), (2, 5, 0)]

            pip1 = Compose([transform1, transform1, transform1])
            kpts_exp = pip1(image=np.zeros((3, 6, 10)), keypoints=kpts_in)['keypoints']
            self.perform_tr(transform3, kpts_in, (3, 6, 10), kpts_exp, print_out=False)

    def test_keypoints_timelapse(self):
        img = np.ones((1, 10, 10, 10, 5))
        kpts = [(0, 0, 0, 0), (3, 3, 3, 0), (7, 6, 5, 0), (0, 0, 0, 2), (3, 3, 3, 2), (7, 6, 5, 2),
                (4, 8, 6, 3), (5, 5, 5, 4), (9, 9, 9, 2)]
        tr_pip = Compose([RandomRotate90([1], factor=1, always_apply=True)])  # y=x    x = h-y
        kpts_exp = [(0, 0, 9, 0), (3, 3, 6, 0), (7, 5, 3, 0), (0, 0, 9, 2), (3, 3, 6, 2), (7, 5, 3, 2),
                    (4, 6, 1, 3), (5, 5, 4, 4), (9, 9, 0, 2)]

        res_dict = tr_pip(image=img, keypoints=kpts)
        kpts_res = res_dict['keypoints']

        self.assertListEqual(kpts_res, kpts_exp)


class TestFlip(unittest.TestCase):
    def test_shape(self):
        tests = get_shape_tests(Flip, (31, 32, 33))
        for tr_img, expected_shape, data_type in tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)

    def test_keypoints(self):
        tests = get_keypoints_tests(Flip, params={})
        for value, expected_value, msg in tests:
            self.assertGreater(value, expected_value * 0.1, msg)

    def perform_tr(self, tr, kpts_in, img_shape_in, kpts_exp, img_shape_exp=None, print_out=False):
        pipeline = Compose([tr])

        img = np.zeros(img_shape_in, dtype=float)

        res_dict = pipeline(image=img, keypoints=kpts_in)

        if img_shape_exp is not None:
            if len(img_shape_in) > 3:
                self.assertTupleEqual(res_dict['image'].shape, img_shape_exp)
            else:
                self.assertTupleEqual(res_dict['image'].shape[1:], img_shape_exp)

        kpts_out = res_dict['keypoints']

        if print_out:
            print(f'exp: {kpts_exp}')
            print(f'out: {kpts_out}')

        for idx in range(len(kpts_exp)):
            self.assertTupleEqual(kpts_out[idx], kpts_exp[idx])

    def test_keypoints_shape_corners(self):
        # around z axis
        transform = Flip(axes=[1], always_apply=True)
        kpts_in = [(0, 0, 0), (0, 0, 9), (0, 5, 9), (0, 5, 0), (2, 0, 0), (2, 0, 9), (2, 5, 9), (2, 5, 0)]
        kpts_exp = [(2, 0, 0), (2, 0, 9), (2, 5, 9), (2, 5, 0), (0, 0, 0), (0, 0, 9), (0, 5, 9), (0, 5, 0)]
        self.perform_tr(transform, kpts_in, (3, 6, 10), kpts_exp, img_shape_exp=(3, 6, 10), print_out=True)

        # around y axis
        transform = Flip(axes=[2], always_apply=True)
        kpts_in = [(0, 0, 0), (0, 0, 9), (0, 5, 9), (0, 5, 0), (2, 0, 0), (2, 0, 9), (2, 5, 9), (2, 5, 0)]
        kpts_exp = [(0, 5, 0), (0, 5, 9), (0, 0, 9), (0, 0, 0), (2, 5, 0), (2, 5, 9), (2, 0, 9), (2, 0, 0)]
        self.perform_tr(transform, kpts_in, (3, 6, 10), kpts_exp, img_shape_exp=(3, 6, 10), print_out=True)

        # around x axis
        transform = Flip(axes=[3], always_apply=True)
        kpts_in = [(0, 0, 0), (0, 0, 9), (0, 5, 9), (0, 5, 0), (2, 0, 0), (2, 0, 9), (2, 5, 9), (2, 5, 0)]
        kpts_exp = [(0, 0, 9), (0, 0, 0), (0, 5, 0), (0, 5, 9), (2, 0, 9), (2, 0, 0), (2, 5, 0), (2, 5, 9)]
        self.perform_tr(transform, kpts_in, (3, 6, 10), kpts_exp, img_shape_exp=(3, 6, 10), print_out=True)

    def test_keypoints_timelapse(self):
        img = np.ones((1, 10, 10, 10, 5))
        kpts = [(0, 0, 0, 0), (3, 3, 3, 0), (7, 6, 5, 0), (0, 0, 0, 2), (3, 3, 3, 2), (7, 6, 5, 2),
                (4, 8, 6, 3), (5, 5, 5, 4), (9, 9, 9, 2)]
        tr_pip = Compose([Flip([1], always_apply=True)])
        kpts_exp = [(9, 0, 0, 0), (6, 3, 3, 0), (2, 6, 5, 0), (9, 0, 0, 2), (6, 3, 3, 2), (2, 6, 5, 2),
                    (5, 8, 6, 3), (4, 5, 5, 4), (0, 9, 9, 2)]

        res_dict = tr_pip(image=img, keypoints=kpts)
        kpts_res = res_dict['keypoints']

        self.assertListEqual(kpts_res, kpts_exp)


class TestRandomFlip(unittest.TestCase):
    def test_shape(self):

        axes_list = [None,
                     [],
                     [1],
                     [1, 2],
                     [1, 2, 3]]

        for _ in range(16):
            for axes in axes_list:
                tests = get_shape_tests(RandomFlip, (30, 30, 30),
                                        params={'axes_to_choose': axes})
                for tr_img, expected_shape, data_type in tests:
                    self.assertTupleEqual(tr_img.shape, expected_shape)
                    self.assertEqual(tr_img.dtype, data_type)

    def test_keypoints(self):

        axes_list = [None,
                     [],
                     [1],
                     [1, 2],
                     [1, 2, 3]]

        for _ in range(16):
            for axes in axes_list:
                tests = get_keypoints_tests(RandomFlip, params={'axes_to_choose': axes})
                for value, expected_value, msg in tests:
                    self.assertGreater(value, expected_value * 0.1, msg)

    def test_no_effect(self):
        for _ in range(50):
            pip = Compose([RandomFlip(axes_to_choose=[])])
            img = np.ones((20, 20, 20))
            res = pip(image=img)['image']
            self.assertTrue(np.all(img == res))


class TestCenterCrop(unittest.TestCase):
    def test_inflate(self):
        in_shape = (32, 31, 30)
        shape_tests = get_shape_tests(CenterCrop, in_shape, {'shape': (40, 41, 43)})

        for tr_img, expected_shape, data_type in shape_tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)

    def test_deflate(self):
        in_shape = (32, 31, 30)
        shape_tests = get_shape_tests(CenterCrop, in_shape, {'shape': (20, 21, 23)})

        for tr_img, expected_shape, data_type in shape_tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)

    def test_keypoints(self):
        in_shape = (32, 31, 30)
        tests = get_keypoints_tests(CenterCrop, in_shape, params={'shape': (40, 41, 42)})
        for value, expected_value, msg in tests:
            self.assertGreater(value, expected_value * 0.5, msg)

        tests = get_keypoints_tests(CenterCrop, in_shape, params={'shape': (20, 21, 22)})
        for value, expected_value, msg in tests:
            self.assertGreater(value, expected_value * 0.5, msg)

    def test_keypoints_timelapse(self):
        img = np.ones((1, 10, 10, 10, 5))
        kpts = [(0, 0, 0, 0), (3, 3, 3, 0), (7, 6, 5, 0), (0, 0, 0, 2), (3, 3, 3, 2), (7, 6, 5, 2),
                (4, 8, 6, 3), (5, 5, 5, 4), (9, 9, 9, 2)]
        tr_pip = Compose([CenterCrop((8, 8, 8), always_apply=True)])
        kpts_exp = [(2, 2, 2, 0), (6, 5, 4, 0), (2, 2, 2, 2), (6, 5, 4, 2), (3, 7, 5, 3), (4, 4, 4, 4)]


        res_dict = tr_pip(image=img, keypoints=kpts)
        kpts_res = res_dict['keypoints']

        self.assertListEqual(kpts_res, kpts_exp)


class TestRandomCrop(unittest.TestCase):
    def test_inflate(self):
        in_shape = (32, 31, 30)
        shape_tests = get_shape_tests(RandomCrop, in_shape, {'shape': (40, 41, 42)})

        for tr_img, expected_shape, data_type in shape_tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)

    def test_deflate(self):
        in_shape = (32, 31, 30)
        shape_tests = get_shape_tests(RandomCrop, in_shape, {'shape': (20, 21, 22)})

        for tr_img, expected_shape, data_type in shape_tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)

    def test_keypoints(self):
        in_shape = (32, 31, 30)
        tests = get_keypoints_tests(RandomCrop, in_shape, params={'shape': (40, 41, 42)})
        for value, expected_value, msg in tests:
            self.assertGreater(value, expected_value * 0.5, msg)

        tests = get_keypoints_tests(RandomCrop, in_shape, params={'shape': (20, 21, 22)})
        for value, expected_value, msg in tests:
            self.assertGreater(value, expected_value * 0.5, msg)


class TestResize(unittest.TestCase):
    def test_inflate(self):
        in_shape = (32, 31, 30)
        shape_tests = get_shape_tests(Resize, in_shape, {'shape': (40, 41, 42)})

        for tr_img, expected_shape, data_type in shape_tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)

    def test_deflate(self):
        in_shape = (32, 31, 30)
        shape_tests = get_shape_tests(Resize, in_shape, {'shape': (20, 21, 22)})

        for tr_img, expected_shape, data_type in shape_tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)

    def test_keypoints(self):
        in_shape = (32, 31, 30)
        tests = get_keypoints_tests(Resize, in_shape, params={'shape': (40, 41, 42)})
        for value, expected_value, msg in tests:
            self.assertGreater(value, expected_value * 0.5, msg)

        tests = get_keypoints_tests(Resize, in_shape, params={'shape': (20, 21, 22)})
        for value, expected_value, msg in tests:
            self.assertGreater(value, expected_value * 0.5, msg)

    def test_keypoints_timelapse(self):
        img = np.ones((1, 10, 10, 10, 5))
        kpts = [(0, 0, 0, 0), (3, 3, 3, 0), (7, 6, 5, 0), (0, 0, 0, 2), (3, 3, 3, 2), (7, 6, 5, 2),
                (4, 8, 6, 3), (5, 5, 5, 4), (9, 9, 9, 2)]
        tr_pip = Compose([Resize((20, 20, 20), always_apply=True)])
        kpts_exp = [(0, 0, 0, 0), (6, 6, 6, 0), (14, 12, 10, 0), (0, 0, 0, 2), (6, 6, 6, 2), (14, 12, 10, 2),
                    (8, 16, 12, 3), (10, 10, 10, 4), (18, 18, 18, 2)]

        res_dict = tr_pip(image=img, keypoints=kpts)
        kpts_res = res_dict['keypoints']

        self.assertListEqual(kpts_res, kpts_exp)


class TestPad(unittest.TestCase):
    def test_1(self):
        tr = Compose([Pad(2)])

        img = np.empty((30, 30, 30))
        tr_img = tr(image=img)['image']
        self.assertTupleEqual(tr_img.shape, (1, 34, 34, 34))

        img = np.empty((1, 30, 30, 30))
        tr_img = tr(image=img)['image']
        self.assertTupleEqual(tr_img.shape, (1, 34, 34, 34))

        img = np.empty((4, 30, 30, 30))
        tr_img = tr(image=img)['image']
        self.assertTupleEqual(tr_img.shape, (4, 34, 34, 34))

        img = np.empty((4, 30, 30, 30, 5))
        tr_img = tr(image=img)['image']
        self.assertTupleEqual(tr_img.shape, (4, 34, 34, 34, 5))

    def test_keypoints(self):
        in_shape = (32, 31, 30)
        tests = get_keypoints_tests(Pad, in_shape, params={'pad_size': (5, 8)})
        for value, expected_value, msg in tests:
            self.assertGreater(value, expected_value * 0.5, msg)

        tests = get_keypoints_tests(Pad, in_shape, params={'pad_size': 4})
        for value, expected_value, msg in tests:
            self.assertGreater(value, expected_value * 0.5, msg)

        tests = get_keypoints_tests(Pad, in_shape, params={'pad_size': (3, 4, 5, 6, 7, 8)})
        for value, expected_value, msg in tests:
            self.assertGreater(value, expected_value * 0.5, msg)

    def test_keypoints_timelapse(self):
        img = np.ones((1, 10, 10, 10, 5))
        kpts = [(0, 0, 0, 0), (3, 3, 3, 0), (7, 6, 5, 0), (0, 0, 0, 2), (3, 3, 3, 2), (7, 6, 5, 2),
                (4, 8, 6, 3), (5, 5, 5, 4), (9, 9, 9, 2)]
        tr_pip = Compose([Pad((0, 0, 1, 0, 3, 0), always_apply=True)])
        kpts_exp = [(0, 1, 3, 0), (3, 4, 6, 0), (7, 7, 8, 0), (0, 1, 3, 2), (3, 4, 6, 2), (7, 7, 8, 2),
                    (4, 9, 9, 3), (5, 6, 8, 4), (9, 10, 12, 2)]

        res_dict = tr_pip(image=img, keypoints=kpts)
        kpts_res = res_dict['keypoints']

        self.assertListEqual(kpts_res, kpts_exp)


class TestRandomAffineTransform(unittest.TestCase):
    def test_shape(self):

        angle_limits = [10,
                        (-20, 20),
                        (12, 30, 0),
                        (-20, 20, -180, 180, 0, 0)]

        for angle_limit in angle_limits:
            tests = get_shape_tests(RandomAffineTransform, (31, 32, 33),
                                    params={'angle_limit': angle_limit})
            for tr_img, expected_shape, data_type in tests:
                self.assertTupleEqual(tr_img.shape, expected_shape)
                self.assertEqual(tr_img.dtype, data_type)

        scale_limits = [0.2,
                        (0.8, 1.2),
                        (0.2, 0.3, 0.1),
                        (0.8, 1.2, 0.9, 1.1, 0.7, 1.)]

        for scale_limit in scale_limits:
            tests = get_shape_tests(RandomAffineTransform, (31, 32, 33),
                                    params={'scaling_limit': scale_limit})
            for tr_img, expected_shape, data_type in tests:
                self.assertTupleEqual(tr_img.shape, expected_shape)
                self.assertEqual(tr_img.dtype, data_type)

        translation_limits = [10,
                              (0, 12),
                              (3, 5, 10),
                              (-3, 3, -5, 5, 0, 0)]

        for translation in translation_limits:
            tests = get_shape_tests(RandomAffineTransform, (31, 32, 33),
                                    params={'translation_limit': translation})
            for tr_img, expected_shape, data_type in tests:
                self.assertTupleEqual(tr_img.shape, expected_shape)
                self.assertEqual(tr_img.dtype, data_type)

    def test_keypoints(self):

        in_shape = (61, 62, 63)

        angle_limits = [10,
                        (-20, 20),
                        (12, 30, 0),
                        (-20, 20, -180, 180, 0, 0)]

        for angle_limit in angle_limits:
            tests = get_keypoints_tests(RandomAffineTransform,
                                        in_shape=in_shape,
                                        params={'angle_limit': angle_limit})

            for value, expected_value, msg in tests:
                self.assertGreater(value, expected_value * 0.1, msg)

        scale_limits = [0.2,
                        (0.8, 1.2),
                        (0.2, 0.3, 0.1),
                        (0.8, 1.2, 0.9, 1.1, 0.7, 1.)]

        for scale_limit in scale_limits:
            tests = get_keypoints_tests(RandomAffineTransform,
                                        in_shape=in_shape,
                                        params={'scaling_limit': scale_limit})

            for value, expected_value, msg in tests:
                self.assertGreater(value, expected_value * 0.5, msg)

        translation_limits = [10,
                              (0, 12),
                              (3, 5, 10),
                              (-3, 3, -5, 5, 0, 0)]

        for translation in translation_limits:
            tests = get_keypoints_tests(RandomAffineTransform,
                                        in_shape=in_shape,
                                        params={'translation_limit': translation})

            for value, expected_value, msg in tests:
                self.assertGreater(value, expected_value * 0.2, msg)


class TestAffineTransform(unittest.TestCase):
    def test_shape(self):

        scale = (1.2, 0.8, 1)
        translation = (0, 1, -40)
        angles = (-20, 0, -0.5)

        tests = get_shape_tests(AffineTransform, (31, 32, 33),
                                params={'translation': translation})
        for tr_img, expected_shape, data_type in tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)

        tests = get_shape_tests(AffineTransform, (31, 32, 33),
                                params={'scale': scale})
        for tr_img, expected_shape, data_type in tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)

        tests = get_shape_tests(AffineTransform, (31, 32, 33),
                                params={'angles': angles})
        for tr_img, expected_shape, data_type in tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)

    def test_keypoints(self):

        scale = (1.2, 0.8, 1)
        translation = (0, 1, -40)
        angles = (-20, 0, -0.5)

        tests = get_keypoints_tests(AffineTransform,
                                    in_shape=(61, 62, 63),
                                    params={'scale': scale})
        for value, expected_value, msg in tests:
            self.assertGreater(value, expected_value * 0.5, msg)

        tests = get_keypoints_tests(AffineTransform,
                                    in_shape=(61, 62, 63),
                                    params={'translation': translation})
        for value, expected_value, msg in tests:
            self.assertGreater(value, expected_value * 0.5, msg)

        tests = get_keypoints_tests(AffineTransform,
                                    in_shape=(61, 62, 63),
                                    params={'angles': angles})
        for value, expected_value, msg in tests:
            self.assertGreater(value, expected_value * 0.5, msg)

    def test_translation_img(self):
        tr = Compose([AffineTransform(translation=(0, -1, 2))])

        img = np.ones((1, 10, 10, 10))
        img[0, 4, 4, 4] = 5

        out = tr(image=img)['image']

        print(out[0,4])
        self.assertEqual(out[0, 4, 4, 4], 1)
        self.assertEqual(out[0, 4, 3, 6], 5)

    def test_rotation_img(self):
        tr = Compose([AffineTransform(angles=(0, 0, 180))])

        img = np.ones((1, 9, 9, 9))
        img[0, 3, 3, 3] = 5

        out = tr(image=img)['image']

        self.assertEqual(out[0, 3, 3, 3], 1)
        self.assertEqual(out[0, 5, 5, 3], 5)

    def test_translation_img_kpts(self):
        tr = Compose([AffineTransform(translation=(0, -1, 2))])

        img = np.ones((1, 10, 10, 10))
        img[0, 4, 4, 4] = 5

        kpts = [(4, 4, 4)]

        out_dict = tr(image=img, keypoints=kpts)
        out_img = out_dict['image']
        out_kpts = out_dict['keypoints']

        self.assertEqual(out_img[0, 4, 4, 4], 1)
        self.assertEqual(out_img[0, 4, 3, 6], 5)

        self.assertEqual(len(out_kpts), 1)
        self.assertTupleEqual(out_kpts[0], (4, 3, 6))

    def test_translation_img_kpts_2(self):
        tr = Compose([AffineTransform(translation=(3, -1, 2))])

        img = np.ones((1, 10, 11, 12))
        img[0, 4, 7, 3] = 5

        kpts = [(4, 7, 3)]

        out_dict = tr(image=img, keypoints=kpts)
        out_img = out_dict['image']
        out_kpts = out_dict['keypoints']

        self.assertEqual(out_img[0, 4, 7, 3], 1)
        self.assertEqual(out_img[0, 7, 6, 5], 5)

        self.assertEqual(len(out_kpts), 1)
        self.assertTupleEqual(out_kpts[0], (7, 6, 5))

    def test_translation_img_kpts_bbox_axiswise(self):
        for tr_vector in [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                          (-2, 0, 0), (0, -2, 0), (0, 0, -2),
                          (3, -6, 3)]:

            tr = Compose([AffineTransform(translation=tr_vector)])

            orig_5_pos = (4, 7, 3)
            orig_5_pos_ch = (0,) + orig_5_pos

            img = np.ones((1, 10, 11, 12))
            img[orig_5_pos_ch] = 5

            kpts = [orig_5_pos]

            bboxes = [[(0, 0, 0), orig_5_pos, 0]]

            out_dict = tr(image=img, keypoints=kpts, bboxes=bboxes)
            out_img = out_dict['image']
            out_kpts = out_dict['keypoints']
            out_bboxes = out_dict['bboxes']

            moved_5_pos = tuple((np.asarray(orig_5_pos) + np.asarray(tr_vector)).tolist())
            moved_5_pos_ch = (0,) + moved_5_pos

            self.assertLessEqual(out_img[orig_5_pos_ch], 1, f'Translation by {tr_vector}, orig pos {orig_5_pos}.')  # 0 in case the original position wa soutside the image domain
            self.assertEqual(out_img[moved_5_pos_ch], 5, f'Translation by {tr_vector}, moved pos {moved_5_pos}.')

            self.assertEqual(len(out_kpts), 1)
            self.assertTupleEqual(out_kpts[0], moved_5_pos, f'Translation by {tr_vector}, moved pos {moved_5_pos}.')

            self.assertEqual(len(out_bboxes), 1)
            self.assertTupleEqual(out_bboxes[0][0], tuple(np.maximum((0, 0, 0), tr_vector).tolist()), f'Translation by {tr_vector}, moved pos {moved_5_pos}.')
            self.assertTupleEqual(out_bboxes[0][1], moved_5_pos, f'Translation by {tr_vector}, moved pos {moved_5_pos}.')

    def test_rotation_img_kpts(self):
        tr = Compose([AffineTransform(angles=(0, 0, 180))])

        img = np.ones((1, 9, 9, 9))
        img[0, 3, 3, 3] = 5

        kpts = [(3, 3, 3)]

        out_dict = tr(image=img, keypoints=kpts)
        out_img = out_dict['image']
        out_kpts = out_dict['keypoints']

        self.assertEqual(out_img[0, 3, 3, 3], 1)
        self.assertEqual(out_img[0, 5, 5, 3], 5)

        self.assertEqual(len(out_kpts), 1)
        self.assertTrue(np.allclose(out_kpts[0], (5, 5, 3)))

    def test_rotation_img_kpts_2(self):
        tr = Compose([AffineTransform(angles=(90, 0, 0))])

        img = np.ones((1, 9, 9, 9))
        img[0, 3, 3, 3] = 5

        kpts = [(3, 3, 3)]

        out_dict = tr(image=img, keypoints=kpts)
        out_img = out_dict['image']
        out_kpts = out_dict['keypoints']

        self.assertEqual(out_img[0, 3, 3, 3], 1)
        self.assertEqual(out_img[0, 3, 3, 5], 5)  # y downwards
        # self.assertEqual(out_img[0, 3, 5, 3], 5)  # y upwards

        self.assertEqual(len(out_kpts), 1)
        self.assertTrue(np.allclose(out_kpts[0], (3, 3, 5)))  # y downwards
        # self.assertTrue(np.allclose(out_kpts[0], (3, 5, 3)))  # y upwards

    def test_rotate_img_kpts_axiswise(self):
        orig_5_pos = (3, 3, 3)
        orig_5_pos_ch = (0,) + orig_5_pos
        for angl, res in [[(90, 0, 0), (3, 3, 5)],  # (3, 3, 5)
                          [(0, 90, 0), (5, 3, 3)],  # (3, 3, 5)
                          [(0, 0, 90), (3, 5, 3)]]:  # (5, 3, 3)

            tr = Compose([AffineTransform(angles=angl)])

            img = np.ones((1, 9, 9, 9))
            img[orig_5_pos_ch] = 5

            kpts = [orig_5_pos]

            out_dict = tr(image=img, keypoints=kpts)
            out_img = out_dict['image']
            out_kpts = out_dict['keypoints']

            moved_5_pos = res
            moved_5_pos_ch = (0,) + moved_5_pos

            self.assertLessEqual(out_img[orig_5_pos_ch], 3, f'Rotation by {angl}, orig pos {orig_5_pos}.')
            self.assertGreaterEqual(out_img[moved_5_pos_ch], 3, f'Rotation by {angl}, moved pos {moved_5_pos}.')

            self.assertEqual(len(out_kpts), 1)
            self.assertTrue(np.allclose(out_kpts[0], moved_5_pos), f'Rotation by {angl}, moved pos {moved_5_pos}.')

    def test_rotate_img_kpts_bbox_axiswise(self):
        orig_5_pos = (3, 3, 3)
        orig_5_pos_ch = (0,) + orig_5_pos
        for angl, res, resbb1, resbb2 in [[(90, 0, 0), (3, 3, 6), (0, 0, 6), (3, 3, 9)],
                                          [(0, 90, 0), (6, 3, 4), (6, 0, 1), (8, 3, 4)],
                                          [(0, 0, 90), (3, 5, 3), (0, 5, 0), (3, 8, 3)]]:

            tr = Compose([AffineTransform(angles=angl)])

            img = np.ones((1, 9, 10, 11))
            img[orig_5_pos_ch] = 5

            kpts = [orig_5_pos]

            bboxes = [[(0, 0, 0), orig_5_pos, 0]]

            out_dict = tr(image=img, keypoints=kpts, bboxes=bboxes)
            out_img = out_dict['image']
            out_kpts = out_dict['keypoints']
            out_bboxes = out_dict['bboxes']

            moved_5_pos = res
            moved_5_pos_ch = (0,) + moved_5_pos

            self.assertLessEqual(out_img[orig_5_pos_ch], 1, f'Rotation by {angl}, orig pos {orig_5_pos}.')
            self.assertGreaterEqual(out_img[moved_5_pos_ch], 2, f'Rotation by {angl}, moved pos {moved_5_pos}.')

            self.assertEqual(len(out_kpts), 1)
            self.assertTrue(np.allclose(out_kpts[0], moved_5_pos, atol=0.6), f'Rotation by {angl}, moved pos {moved_5_pos}, out_kpt {out_kpts[0]}.')  # all within half a pixel

            self.assertEqual(len(out_bboxes), 1)
            self.assertTrue(np.allclose(out_bboxes[0][0], resbb1, atol=0.6), f'Rotation by {angl}, moved pos {resbb1}, TL corner {out_bboxes[0][0]}.')  # all within half a pixel
            self.assertTrue(np.allclose(out_bboxes[0][1], resbb2, atol=0.6), f'Rotation by {angl}, moved pos {resbb2}, BR corner {out_bboxes[0][1]}.')  # all within half a pixel

    def test_scale_img_kpts(self):
        tr = Compose([AffineTransform(scale=(1, 1, 2))])

        img = np.ones((1, 9, 9, 9))
        img[0, 3, 3, 3] = 5

        kpts = [(3, 3, 3)]

        out_dict = tr(image=img, keypoints=kpts)
        out_img = out_dict['image']
        out_kpts = out_dict['keypoints']

        self.assertNotEqual(out_img[0, 3, 3, 3], 5)
        self.assertEqual(out_img[0, 3, 3, 2], 5)

        self.assertEqual(len(out_kpts), 1)
        self.assertTrue(np.allclose(out_kpts[0], (3, 3, 2)))

    def test_scale_img_kpts_axiswise(self):
        for scales in [(2, 1, 1), (1, 2, 1), (1, 1, 2)]:

            tr = Compose([AffineTransform(scale=scales)])

            orig_5_pos = (3, 3, 3)
            orig_5_pos_ch = (0,) + orig_5_pos

            img = np.ones((1, 9, 9, 9))
            img[orig_5_pos_ch] = 5

            kpts = [orig_5_pos]

            out_dict = tr(image=img, keypoints=kpts)
            out_img = out_dict['image']
            out_kpts = out_dict['keypoints']

            moved_5_pos = tuple((np.asarray(orig_5_pos) * np.asarray(scales) - (np.asarray(scales) - 1) * (9 // 2)).tolist())
            moved_5_pos_ch = (0,) + moved_5_pos

            self.assertLessEqual(out_img[orig_5_pos_ch], 3)
            self.assertGreaterEqual(out_img[moved_5_pos_ch], 3)

            self.assertEqual(len(out_kpts), 1)
            self.assertTupleEqual(out_kpts[0], moved_5_pos)

    def test_scale_img_kpts_axiswise_2(self):
        for scales in [(0.5, 1, 1), (1, 0.5, 1), (1, 1, 0.5)]:
            tr = Compose([AffineTransform(scale=scales)])

            orig_5_pos = (2, 2, 2)
            orig_5_pos_ch = (0,) + orig_5_pos

            img = np.ones((1, 9, 9, 9))
            img[orig_5_pos_ch] = 5

            kpts = [orig_5_pos]

            out_dict = tr(image=img, keypoints=kpts)
            out_img = out_dict['image']
            out_kpts = out_dict['keypoints']

            moved_5_pos = tuple((np.asarray(orig_5_pos) * np.asarray(scales) + (np.asarray(scales) < 1) * (9 // 4)).tolist())
            moved_5_pos_ch = (0,) + tuple(np.round(np.asarray(moved_5_pos)).astype(int).tolist())

            self.assertLessEqual(out_img[orig_5_pos_ch], 3, f'Scale by {scales}, orig pos {orig_5_pos}.')
            self.assertGreaterEqual(out_img[moved_5_pos_ch], 3, f'Scale by {scales}, moved pos {moved_5_pos}.')

            self.assertEqual(len(out_kpts), 1)
            self.assertTupleEqual(out_kpts[0], moved_5_pos, f'Scale by {scales}, moved pos {moved_5_pos}.')

    def test_scale_img_kpts_bbox_axiswise(self):
        for scales in [(2, 1, 1), (1, 2, 1), (1, 1, 2)]:

            tr = Compose([AffineTransform(scale=scales)])

            orig_5_pos = (3, 3, 3)
            orig_5_pos_ch = (0,) + orig_5_pos

            img = np.ones((1, 10, 10, 10))
            img[orig_5_pos_ch] = 5

            kpts = [orig_5_pos]

            bboxes = [[(0, 0, 0), orig_5_pos, 0]]

            out_dict = tr(image=img, keypoints=kpts, bboxes=bboxes)
            out_img = out_dict['image']
            out_kpts = out_dict['keypoints']
            out_bboxes = out_dict['bboxes']

            moved_5_pos = tuple((np.asarray(orig_5_pos) * np.asarray(scales) - (np.asarray(scales) - 1) * 4.5).tolist())
            moved_5_pos_ch = (0,) + tuple(np.round(moved_5_pos).astype(int).tolist())

            self.assertLessEqual(out_img[orig_5_pos_ch], 3)
            self.assertGreaterEqual(out_img[moved_5_pos_ch], 3)

            self.assertEqual(len(out_kpts), 1)
            self.assertTupleEqual(out_kpts[0], moved_5_pos)

            self.assertEqual(len(out_bboxes), 1)
            self.assertTupleEqual(out_bboxes[0][0], (0, 0, 0))
            self.assertTupleEqual(out_bboxes[0][1], moved_5_pos)

    def test_scale_img_kpts_bbox_axiswise_2(self):
        for scales in [(2, 1, 1), (1, 2, 1), (1, 1, 2)]:

            tr = Compose([AffineTransform(scale=scales)])

            orig_5_pos = (3, 3, 3)
            orig_5_pos_ch = (0,) + orig_5_pos

            img = np.ones((1, 10, 11, 12))
            img[orig_5_pos_ch] = 5

            kpts = [orig_5_pos]

            bboxes = [[(0, 0, 0), orig_5_pos, 0]]

            out_dict = tr(image=img, keypoints=kpts, bboxes=bboxes)
            out_img = out_dict['image']
            out_kpts = out_dict['keypoints']
            out_bboxes = out_dict['bboxes']

            moved_5_pos = tuple((np.asarray(orig_5_pos) * np.asarray(scales) - (np.asarray(scales) - 1) * np.asarray([4.5, 5, 5.5])).tolist())
            moved_5_pos_ch = (0,) + tuple(np.round(moved_5_pos).astype(int).tolist())

            self.assertLessEqual(out_img[orig_5_pos_ch], 3)
            self.assertGreaterEqual(out_img[moved_5_pos_ch], 3)

            self.assertEqual(len(out_kpts), 1)
            self.assertTupleEqual(out_kpts[0], moved_5_pos)

            self.assertEqual(len(out_bboxes), 1)
            self.assertTupleEqual(out_bboxes[0][0], (0, 0, 0))
            self.assertTupleEqual(out_bboxes[0][1], moved_5_pos)

    def test_keypoints_timelapse(self):
        img = np.ones((1, 10, 10, 10, 5))
        kpts = [(0, 0, 0, 0), (3, 3, 3, 0), (7, 6, 5, 0), (0, 0, 0, 2), (3, 3, 3, 2), (7, 6, 5, 2),
                (4, 8, 6, 3), (5, 5, 5, 4), (9, 9, 9, 2)]
        tr_pip = Compose([AffineTransform(translation=(0, 0, 1), always_apply=True)])
        kpts_exp = [(0, 0, 1, 0), (3, 3, 4, 0), (7, 6, 6, 0), (0, 0, 1, 2), (3, 3, 4, 2), (7, 6, 6, 2),
                    (4, 8, 7, 3), (5, 5, 6, 4)]

        res_dict = tr_pip(image=img, keypoints=kpts)
        kpts_res = res_dict['keypoints']

        self.assertListEqual(kpts_res, kpts_exp)


# ImageTransforms
class TestNormalizeMeanStd(unittest.TestCase):
    def test_shape(self):
        mean = 1.2
        std = 2
        tests = get_shape_tests(NormalizeMeanStd, (31, 32, 33),
                                params={'mean': mean,
                                        'std': std})
        for tr_img, expected_shape, data_type in tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)


class TestGaussianNoise(unittest.TestCase):
    def test_shape(self):
        tests = get_shape_tests(GaussianNoise, (31, 32, 33))
        for tr_img, expected_shape, data_type in tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)


class TestPoissonNoise(unittest.TestCase):
    def test_shape(self):
        tests = get_shape_tests(PoissonNoise, (31, 32, 33))
        for tr_img, expected_shape, data_type in tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)


class TestGaussianBlur(unittest.TestCase):
    def test_shape(self):
        tests = get_shape_tests(GaussianBlur, (31, 32, 33))
        for tr_img, expected_shape, data_type in tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)

    def test_shape_5d(self):
        for params in [{'sigma': 1}, {'sigma': (1, 2, 2)}, {'sigma': (1, 2, 2, 3)},
                       {'sigma': [1, 2]}, {'sigma': [(1, 2, 2), (1, 2, 2)]}, {'sigma': [(1, 2, 2, 3), (1, 2, 2, 3)]}]:
            tests = get_shape_tests_5d(GaussianBlur, (2, 31, 32, 33, 5), params=params)
            for tr_img, expected_shape, data_type in tests:
                self.assertTupleEqual(tr_img.shape, expected_shape)
                self.assertEqual(tr_img.dtype, data_type)


class TestRandomGaussianBlur(unittest.TestCase):
    def test_shape(self):
        tests = get_shape_tests(RandomGaussianBlur, (31, 32, 33))
        for tr_img, expected_shape, data_type in tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)

    def test_shape_5d(self):
        for params in [{'max_sigma': 1}, {'max_sigma': (1, 2, 2)}, {'max_sigma': (1, 2, 2, 3)},
                       {'max_sigma': [1, 2]}, {'max_sigma': [(1, 2, 2), (1, 2, 2)]},
                       {'max_sigma': [(1, 2, 2, 3), (1, 2, 2, 3)]}]:
            tests = get_shape_tests_5d(RandomGaussianBlur, (2, 31, 32, 33, 5), params=params)
            for tr_img, expected_shape, data_type in tests:
                self.assertTupleEqual(tr_img.shape, expected_shape)
                self.assertEqual(tr_img.dtype, data_type)


class TestRandomGamma(unittest.TestCase):
    def test_shape(self):
        tests = get_shape_tests(RandomGamma, (31, 32, 33))
        for tr_img, expected_shape, data_type in tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)


class TestRandomBrightnessContrast(unittest.TestCase):
    def test_shape(self):

        brightness_list = [3, (2, 5)]
        contrast_list = [0.5, (.7, 1.1)]

        for brightness in brightness_list:
            for contrast in contrast_list:
                tests = get_shape_tests(RandomBrightnessContrast, (30, 31, 32),
                                        params={'brightness_limit': brightness,
                                                'contrast_limit': contrast})
                for tr_img, expected_shape, data_type in tests:
                    self.assertTupleEqual(tr_img.shape, expected_shape)
                    self.assertEqual(tr_img.dtype, data_type)


class TestHistogramEqualization(unittest.TestCase):
    def test_shape(self):
        tests = get_shape_tests(HistogramEqualization, (31, 32, 33))
        for tr_img, expected_shape, data_type in tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)


class TestNormalize(unittest.TestCase):
    def test_shape(self):
        tests = get_shape_tests(Normalize, (31, 32, 33))
        for tr_img, expected_shape, data_type in tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)


class TestRescale(unittest.TestCase):
    def test_shape(self):
        in_shape = (31, 32, 33)
        scale = 2
        tests = get_shape_tests(Rescale, in_shape, params={'scales': scale},
                                exp_shape=np.asarray(in_shape) * scale)
        for tr_img, expected_shape, data_type in tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)

        in_shape = (30, 33, 60)
        scale = 1.0 / 3
        tests = get_shape_tests(Rescale, in_shape, params={'scales': scale},
                                exp_shape=np.asarray(in_shape) * scale)
        for tr_img, expected_shape, data_type in tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)

        in_shape = (30, 33, 60)
        scale = (0.5, 3, 1.5)
        tests = get_shape_tests(Rescale, in_shape, params={'scales': scale},
                                exp_shape=np.asarray(in_shape) * np.asarray(scale))
        for tr_img, expected_shape, data_type in tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)

    def test_keypoints(self):
        in_shape = (32, 31, 30)
        scale = 2
        tests = get_keypoints_tests(Rescale, in_shape, params={'scales': scale})
        for value, expected_value, msg in tests:
            self.assertGreater(value, expected_value * 0.5, msg)

        in_shape = (30, 33, 60)
        scale = (0.5, 3, 1.5)
        tests = get_keypoints_tests(Rescale, in_shape, params={'scales': scale})
        for value, expected_value, msg in tests:
            self.assertGreater(value, expected_value * 0.5, msg)


class TestRemoveBackgroundGaussian(unittest.TestCase):
    def test_shape(self):
        tests = get_shape_tests(RemoveBackgroundGaussian, (31, 32, 33))
        for tr_img, expected_shape, data_type in tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)

    def test_shape_5d(self):
        for params in [{'sigma': 1}, {'sigma': (1, 2, 2)}, {'sigma': (1, 2, 2, 3)},
                       {'sigma': [1, 2]}, {'sigma': [(1, 2, 2), (1, 2, 2)]}, {'sigma': [(1, 2, 2, 3), (1, 2, 2, 3)]}]:
            tests = get_shape_tests_5d(RemoveBackgroundGaussian, (2, 31, 32, 33, 5), params=params)
            for tr_img, expected_shape, data_type in tests:
                self.assertTupleEqual(tr_img.shape, expected_shape)
                self.assertEqual(tr_img.dtype, data_type)


class TestInputArgs(unittest.TestCase):
    def test_individual_transforms(self):
        tr = Compose([
            Resize((20, 30, 40)),
            Scale(0.8), Scale((0.9, 0.3, 1.2)),
            RandomScale(0.5), RandomScale((0.5, 0.8)), RandomScale((0.2, 0.5, 1.1)),
            RandomScale((0.2, 0.4, 0.8, 0.9, 1.1, 1.2)),
            RandomRotate90([1]), RandomRotate90([1, 2, 3]), RandomRotate90(None), RandomRotate90([1, 1, 1]),
            Flip([1]), Flip([1, 2, 3]), Flip(None), Flip([1, 1, 1]),
            RandomFlip([1, 2]), RandomFlip(None), RandomFlip([]),
            CenterCrop((20, 30, 40)),
            RandomCrop((20, 30, 40)),
            RandomAffineTransform(angle_limit=45), RandomAffineTransform(angle_limit=(45, 60)),
            RandomAffineTransform(angle_limit=(45, 60, 90)),
            RandomAffineTransform(angle_limit=(30, 35, 50, 60, 80, 90)),
            RandomAffineTransform(translation_limit=45), RandomAffineTransform(translation_limit=(45, 60)),
            RandomAffineTransform(translation_limit=(45, 60, 90)),
            RandomAffineTransform(translation_limit=(30, 35, 50, 60, 80, 90)),
            RandomAffineTransform(scaling_limit=0.45), RandomAffineTransform(scaling_limit=(0.45, 0.60)),
            RandomAffineTransform(scaling_limit=(0.45, 0.60, 0.90)),
            RandomAffineTransform(scaling_limit=(0.30, 0.35, 0.50, 0.60, 0.80, 0.90)),
            AffineTransform(angles=(45, 60, 90), translation=(45, 60, 90), scale=(0.5, 0.8, 0.8)),
            GaussianNoise((0.3, 0.5), 8),
            PoissonNoise((0.3, 0.5)),
            NormalizeMeanStd(3, 4), NormalizeMeanStd((3, 4), (3, 4)),
            GaussianBlur(1), GaussianBlur((1, 2, 2)), GaussianBlur((1, 2, 2, 3)),
            GaussianBlur([1, 2]), GaussianBlur([(1, 2, 2), (1, 2, 2)]),
            GaussianBlur([(1, 2, 2, 3), (1, 2, 2, 3)]),
            RandomGaussianBlur(1), RandomGaussianBlur((1, 2, 2)), RandomGaussianBlur((1, 2, 2, 3)),
            RandomGaussianBlur([1, 2]), RandomGaussianBlur([(1, 2, 2), (1, 2, 2)]),
            RandomGaussianBlur([(1, 2, 2, 3), (1, 2, 2, 3)]),
            RandomGamma((0.5, 0.9)),
            RandomBrightnessContrast(1, 1), RandomBrightnessContrast((1, 2), (1, 3)),
            RandomBrightnessContrast(1, (2, 3)),
            HistogramEqualization(30),
            Pad(10), Pad((10, 30)), Pad((10, 20, 40, 15, 20, 20)),
            Normalize(2, 4), Normalize([1, 2], [1, 3]),
            Rescale(0.8), Rescale((0.9, 0.3, 1.2)),
            RemoveBackgroundGaussian(1), RemoveBackgroundGaussian((1, 2, 2)), RemoveBackgroundGaussian((1, 2, 2, 3)),
            RemoveBackgroundGaussian([1, 2]), RemoveBackgroundGaussian([(1, 2, 2), (1, 2, 2)]),
            RemoveBackgroundGaussian([(1, 2, 2, 3), (1, 2, 2, 3)]),
        ])

    def test_individual_transforms_incorrect_initialisation(self):
        with self.assertWarns(Warning):
            tr = Compose([Resize((30, 40))])

        with self.assertWarns(Warning):
            tr = Compose([Resize((30, 40, 20, 20))])

        with self.assertRaises(BaseException):
            tr = Compose([Scale((30, 40))])

        with self.assertRaises(BaseException):
            tr = Compose([Scale((30, 40, 20, 20))])

        with self.assertWarns(Warning):
            tr = Compose([RandomFlip([1, 3, 0])])

        with self.assertWarns(Warning):
            tr = Compose([Flip([1, 3, 0])])

        with self.assertWarns(Warning):
            tr = Compose([CenterCrop((30, 40))])

        with self.assertWarns(Warning):
            tr = Compose([CenterCrop((30, 40, 0))])

        with self.assertWarns(Warning):
            tr = Compose([CenterCrop((30, 40, 20, 20))])

        with self.assertWarns(Warning):
            tr = Compose([RandomCrop((30, 40))])

        with self.assertWarns(Warning):
            tr = Compose([RandomCrop((30, 40, 0))])

        with self.assertWarns(Warning):
            tr = Compose([RandomCrop((30, 40, 20, 20))])

        with self.assertRaises(BaseException):
            tr = Compose([NormalizeMeanStd(3, (4, 5))])

    def test_individual_transforms_incorrect_argument_values(self):
        with self.assertWarns(Warning):
            tr = Compose([Normalize(2, [3, 4])])
            res = tr(image=np.ones((20, 20, 20)))


class TestInvalidInput(unittest.TestCase):
    def invalid_range_check(self, transform, sample=None, **params):
        if sample is None:
            img_shape = (4, 120, 120, 120)
            img = np.ones(img_shape, dtype=np.float64)
            mask = np.ones(img_shape[1:], dtype=np.int64)
            fmask = np.ones(img_shape[1:], dtype=np.float64)
        else:
            img, mask, fmask = sample

        tr = Compose([transform(p=1, **params)])

        tr_img = tr(image=img, mask=mask, float_mask=fmask)

        # some checks - we just need to make sure that the computation did not fail
        self.assertTrue(np.issubdtype(tr_img['image'].dtype, np.floating))
        self.assertTrue(np.issubdtype(tr_img['mask'].dtype, np.integer))
        self.assertTrue(np.issubdtype(tr_img['float_mask'].dtype, np.floating))

    def test_invalid_range_crop(self):
        self.invalid_range_check(RandomCrop, shape=(10, 10, 10))
        self.invalid_range_check(CenterCrop, shape=(10, 10, 10))

    def test_invalid_range_scale(self):
        self.invalid_range_check(Scale, scales=0.5)

    def test_invalid_range_gamma(self):
        img_shape = (4, 120, 120, 120)
        img = np.ones(img_shape, dtype=np.float64) * 2
        mask = np.ones(img_shape[1:], dtype=np.int64)
        fmask = np.ones(img_shape[1:], dtype=np.float64)
        self.invalid_range_check(RandomGamma, sample=(img, mask, fmask))

    def test_invalid_range_gaussian_blur(self):
        self.invalid_range_check(GaussianBlur)

    def test_invalid_range_normalize(self):
        img_shape = (4, 120, 120, 120)
        img = np.ones(img_shape, dtype=np.float64)
        mask = np.ones(img_shape[1:], dtype=np.int64)
        fmask = np.ones(img_shape[1:], dtype=np.float64)
        self.invalid_range_check(RandomGamma, sample=(img, mask, fmask))

    def invalid_dtype_check(self, transform, **params):
        img_shape = (4, 120, 120, 120)
        img = np.ones(img_shape, dtype=int)
        mask = np.ones(img_shape[1:], dtype=float)
        fmask = np.ones(img_shape[1:], dtype=int)

        tr = Compose([transform(p=1, **params)])

        tr_img = tr(image=img, mask=mask, float_mask=fmask)

        self.assertTrue(np.issubdtype(tr_img['image'].dtype, np.floating))
        self.assertTrue(np.issubdtype(tr_img['mask'].dtype, np.integer))
        self.assertTrue(np.issubdtype(tr_img['float_mask'].dtype, np.floating))

    def test_invalid_dtype_crop(self):
        self.invalid_dtype_check(RandomCrop, shape=(10, 10, 10))
        self.invalid_dtype_check(CenterCrop, shape=(10, 10, 10))

    def test_invalid_dtype_scale(self):
        self.invalid_dtype_check(Scale, scales=0.5)

    def test_invalid_dtype_gamma(self):
        self.invalid_dtype_check(RandomGamma)

    def test_invalid_dtype_gaussian_blur(self):
        self.invalid_dtype_check(RandomGamma)

    def test_invalid_size_crop(self):
        img_shape = (4, 120, 120, 120)
        img = np.ones(img_shape, dtype=np.float64)
        mask = np.ones(img_shape[1:], dtype=np.int64)
        fmask = np.ones(img_shape[1:], dtype=np.float64)

        tr = Compose([CenterCrop(shape=(140, 120, 100), p=1)])

        tr_img = tr(image=img, mask=mask, float_mask=fmask)

        # some checks - we just need to make sure that the computation did not fail
        self.assertTrue(np.issubdtype(tr_img['image'].dtype, np.floating))
        self.assertTrue(np.issubdtype(tr_img['mask'].dtype, np.integer))
        self.assertTrue(np.issubdtype(tr_img['float_mask'].dtype, np.floating))


class TestAlwaysapplyP(unittest.TestCase):
    def test_1(self):
        img = np.random.randn(1, 10, 20, 30)

        tr_pip = Compose([Rescale(scales=(2, 2, 2), p=1)])
        for _ in range(500):
            res = tr_pip(image=img)['image']
            self.assertTupleEqual((1, 20, 40, 60), res.shape)

        tr_pip = Compose([Rescale(scales=(2, 2, 2), p=0)])
        for _ in range(500):
            res = tr_pip(image=img)['image']
            self.assertTupleEqual((1, 10, 20, 30), res.shape)

        tr_pip = Compose([Rescale(scales=(2, 2, 2), p=0.5)])
        applied_count = 0
        for _ in range(500):
            res = tr_pip(image=img)['image']
            if res.shape == (1, 20, 40, 60):
                applied_count += 1
        self.assertTrue(applied_count > 0)
        self.assertTrue(applied_count < 500)


if __name__ == '__main__':
    unittest.main()
