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

from bio_volumentations.augmentations.transforms import (
    GaussianNoise, PoissonNoise, Resize, Pad, Scale, Flip, CenterCrop, AffineTransform,
    RandomScale, RandomRotate90, RandomFlip, RandomCrop, RandomAffineTransform, RandomGamma,
    NormalizeMeanStd, GaussianBlur, Normalize, HistogramEqualization, RandomBrightnessContrast,
    RandomGaussianBlur)

from bio_volumentations.core.composition import Compose
import numpy as np
from typing import Any

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
        keypoints.append((w1-0., h1-0., d1-0.))

    sample = {'image': img,
              'mask': mask,
              'keypoints': keypoints}

    tr = Compose([transform(**params, p=1)])
    sample_transformed = tr(**sample)

    keypoints_transformed = sample_transformed['keypoints']
    if DEBUG:
        print('KEYPOINTS', transform, keypoints)
        print('KEYPOINTS TRANSFORMED', transform, keypoints_transformed)

    tests = []
    for k in keypoints_transformed:
        coos = (np.array(k) + .5).astype(int)
        tests.append((sample_transformed['image'][0, coos[0], coos[1], coos[2]], 10.,
                      f'mask, {k} {coos} {transform} {params}'))
        tests.append((sample_transformed['mask'][coos[0], coos[1], coos[2]], 10.,
                      f'img {k} {coos} {transform}, {params}'))

    return tests


def get_shape_tests(transform,
                    in_shape: tuple,
                    params={}):
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
    w_, h_, d_ = params['shape'] if 'shape' in params.keys() else (w, h, d)

    res = []
    tr = Compose([transform(**params, p=1)])

    # img (W, H, D), mask (W, H, D)
    img = np.ones((w, h, d), dtype=np.float32)
    mask = np.ones((w, h, d), dtype=np.int32)
    fmask = np.ones((w, h, d), dtype=np.float32)
    #print(img.dtype, mask.dtype, fmask.dtype)
    tr_img = tr(image=img, mask=mask, float_mask=fmask)
    #print(tr_img['image'].dtype, tr_img['mask'].dtype, tr_img['float_mask'].dtype)
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

# Helper functions for testing image values and keypoints
def create_test_img(shape: tuple, coordinates: list[tuple]):
    """
    Creates a binary test image with dimensions [C, Z, Y, X, T]
    Args:
        shape: dimensions of the image (C, Z, Y, X, T)
        coordinates: (Z, Y, X, T) points that will be set to 1

    Returns:
        binary test image of the given shape
    """
    img = np.zeros(shape)
    for coords in coordinates:
        z, y, x, t = coords
        img[0][z][y][x][t] = 1

    return img


def evaluate_result(sample: dict[str, Any], expected_img, expected_keypoints: list[tuple[int, int, int, int]]):
    """
    Evaluates the result image and list of keypoints with the expected values
    Args:
        sample: dictionary containing keys image and keypoints with their respective values
        expected_img: expected image
        expected_keypoints: list of expected keypoints

    Returns:
        result of evaluation
    """
    assert len(expected_keypoints) == len(sample["keypoints"])

    if not np.array_equal(expected_img, sample["image"]):
        return False

    for expected_kp, kp in zip(expected_keypoints, sample["keypoints"]):
        if not np.array_equal(expected_kp, kp):
            return False
    return True


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

        axes_list = [None,
                     [1],
                     [1, 2],
                     [1, 2, 3]]

        for axes in axes_list:
            tests = get_shape_tests(RandomRotate90, (30, 30, 30),
                                    params={'axes': axes})
            for tr_img, expected_shape, data_type in tests:
                self.assertTupleEqual(tr_img.shape, expected_shape)
                self.assertEqual(tr_img.dtype, data_type)

    def test_keypoints(self):

        axes_list = [None,
                     [1],
                     [1, 2],
                     [1, 2, 3]]

        for _ in range(32):
            for axes in axes_list:
                tests = get_keypoints_tests(RandomRotate90, params={'axes': axes})
                for value, expected_value, msg in tests:
                    self.assertGreater(value, expected_value * 0.1, msg)


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

    def test_temporal_only(self):
        img = create_test_img((1, 2, 2, 2, 2), [(0, 0, 0, 0)])
        keypoints = [(0, 0, 0, 0)]

        t_flip = Compose([Flip(axes=[], temporal_apply=True)])
        flipped_sample = t_flip(image=img, keypoints=keypoints)
        expected_img = create_test_img((1, 2, 2, 2, 2), [(0, 0, 0, 1)])
        expected_keypoints = [(0, 0, 0, 1)]
        self.assertTrue(evaluate_result(flipped_sample, expected_img, expected_keypoints))

    def test_temporal_and_x(self):
        img = create_test_img((1, 2, 2, 2, 2), [(0, 0, 0, 0)])
        keypoints = [(0, 0, 0, 0)]

        tx_flip = Compose([Flip(axes=[3], temporal_apply=True)])
        flipped_sample = tx_flip(image=img, keypoints=keypoints)
        expected_img = create_test_img((1, 2, 2, 2, 2), [(0, 0, 1, 1)])
        expected_keypoints = [(0, 0, 1, 1)]
        self.assertTrue(evaluate_result(flipped_sample, expected_img, expected_keypoints))

    def test_temporal_and_xy(self):
        img = create_test_img((1, 2, 2, 2, 2), [(0, 0, 0, 0)])
        keypoints = [(0, 0, 0, 0)]

        txy_flip = Compose([Flip(axes=[2, 3], temporal_apply=True)])
        flipped_sample = txy_flip(image=img, keypoints=keypoints)
        expected_img = create_test_img((1, 2, 2, 2, 2), [(0, 1, 1, 1)])
        expected_keypoints = [(0, 1, 1, 1)]
        self.assertTrue(evaluate_result(flipped_sample, expected_img, expected_keypoints))

    def test_temporal_and_xyz(self):
        img = create_test_img((1, 2, 2, 2, 2), [(0, 0, 0, 0)])
        keypoints = [(0, 0, 0, 0)]

        txyz_flip = Compose([Flip(axes=[1, 2, 3], temporal_apply=True)])
        flipped_sample = txyz_flip(image=img, keypoints=keypoints)
        expected_img = create_test_img((1, 2, 2, 2, 2), [(1, 1, 1, 1)])
        expected_keypoints = [(1, 1, 1, 1)]
        self.assertTrue(evaluate_result(flipped_sample, expected_img, expected_keypoints))


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


    def test_temporal(self ):
        img = create_test_img((1, 2, 2, 2, 2), [(0, 0, 0, 0)])
        keypoints = [(0, 0, 0, 0)]

        rflip = Compose([RandomFlip(axes_to_choose=[1, 2, 3], temporal_apply=True)])
        flipped_sample = rflip(image=img, keypoints=keypoints)
        coordinates = [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1),
                       (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1),
                       (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 0, 1, 1),
                       (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 1, 1)]
        res = False
        for coords in coordinates:
            expected_img = create_test_img((1, 2, 2, 2, 2), [coords])
            expected_keypoint = coords
            if evaluate_result(flipped_sample, expected_img, [expected_keypoint]):
                # fits one of the possibilities
                self.assertTrue(True)
                return
        self.assertTrue(False)


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

    def test_temporal_only(self):
        img = np.ones((1, 1, 1, 1, 1))
        keypoints = [(0, 0, 0, 0)]
        p1, p2 = 1, 2

        t_pad = Compose([Pad(pad_size=0, temporal_pad_size=(p1, p2))])
        padded_sample = t_pad(image=img, keypoints=keypoints)

        expected_img = create_test_img((1, 1, 1, 1,  p1 + 1 + p2), [(0, 0, 0, 1)])
        expected_keypoints = [(0, 0, 0, 1)]
        self.assertTrue(evaluate_result(padded_sample, expected_img, expected_keypoints))

    def test_temporal_x(self):
        img = np.ones((1, 1, 1, 1, 1))
        keypoints = [(0, 0, 0, 0)]
        p1, p2 = 1, 2

        tx_pad = Compose([Pad(pad_size=(0, 0, 0, 0, 1, 2), temporal_pad_size=(p1, p2))])
        padded_sample = tx_pad(image=img, keypoints=keypoints)

        expected_img = create_test_img((1, 1, 1,  p1 + 1 + p2,  p1 + 1 + p2), [(0, 0, 1, 1)])
        expected_keypoints = [(0, 0, 1, 1)]
        self.assertTrue(evaluate_result(padded_sample, expected_img, expected_keypoints))


    def test_temporal_xy(self):
        img = np.ones((1, 1, 1, 1, 1))
        keypoints = [(0, 0, 0, 0)]
        p1, p2 = 1, 2

        txy_pad = Compose([Pad(pad_size=(0, 0, 1, 2, 1, 2), temporal_pad_size=(p1, p2))])
        padded_sample = txy_pad(image=img, keypoints=keypoints)
        expected_img = create_test_img((1, 1,  p1 + 1 + p2,  p1 + 1 + p2, p1 + 1 + p2), [(0, 1, 1, 1)])
        expected_keypoints = [(0, 1, 1, 1)]
        self.assertTrue(evaluate_result(padded_sample, expected_img, expected_keypoints))

    def test_temporal_xyz(self):
        img = np.ones((1, 1, 1, 1, 1))
        keypoints = [(0, 0, 0, 0)]
        p1, p2 = 1, 2

        txyz_pad = Compose([Pad(pad_size=(1, 2, 1, 2, 1, 2), temporal_pad_size=(p1, p2))])
        padded_sample = txyz_pad(image=img, keypoints=keypoints)
        expected_img = create_test_img((1,  p1 + 1 + p2,  p1 + 1 + p2,  p1 + 1 + p2,  p1 + 1 + p2),
                                       [(1, 1, 1, 1)])
        expected_keypoints = [(1, 1, 1, 1)]
        self.assertTrue(evaluate_result(padded_sample, expected_img, expected_keypoints))


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

'''
class TestRescale(unittest.TestCase):
    def test_shape(self):
        in_shape = (31, 32, 33)
        scale = 2
        tests = get_shape_tests(Rescale, in_shape, params={'scales': scale, 'shape': np.asarray(in_shape) * scale})
        for tr_img, expected_shape, data_type in tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)

        in_shape = (30, 33, 60)
        scale = 1.0/3
        tests = get_shape_tests(Rescale, in_shape, params={'scales': scale, 'shape': np.asarray(in_shape) * scale})
        for tr_img, expected_shape, data_type in tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)

        in_shape = (30, 33, 60)
        scale = (0.5, 3, 1.5)
        tests = get_shape_tests(Rescale, in_shape, params={'scales': scale,
                                                           'shape': np.asarray(in_shape) * np.asarray(scale)})
        for tr_img, expected_shape, data_type in tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)

    def test_keypoints(self):
        in_shape = (32, 31, 30)
        scale = 2
        tests = get_keypoints_tests(Rescale, in_shape, params={'scales': scale, 'shape': np.asarray(in_shape) * scale})
        for value, expected_value, msg in tests:
            self.assertGreater(value, expected_value * 0.5, msg)

        in_shape = (30, 33, 60)
        scale = (0.5, 3, 1.5)
        tests = get_keypoints_tests(Rescale, in_shape, params={'scales': scale,
                                                               'shape': np.asarray(in_shape) * np.asarray(scale)})
        for value, expected_value, msg in tests:
            self.assertGreater(value, expected_value * 0.5, msg)
'''

'''
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
'''

class TestInputArgs(unittest.TestCase):
    def test_individual_transforms(self):
        tr = Compose([
            Resize((20, 30, 40)),
            Scale(0.8), Scale((0.9, 0.3, 1.2)),
            RandomScale(0.5), RandomScale((0.5, 0.8)), RandomScale((0.2, 0.5, 1.1)),
            RandomScale((0.2, 0.4, 0.8, 0.9, 1.1, 1.2)),
            RandomRotate90([1]), RandomRotate90([1, 2, 3]), RandomRotate90(None), RandomRotate90([1, 1, 1]),
            Flip([1]), Flip([1, 2, 3]), Flip(None), Flip([1, 1, 1]),
            RandomFlip([(1,), (2, 3)]), RandomFlip(None),
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
            # Rescale(0.8), Rescale((0.9, 0.3, 1.2)),
           # RemoveBackgroundGaussian(1), RemoveBackgroundGaussian((1, 2, 2)), RemoveBackgroundGaussian((1, 2, 2, 3)),
            #RemoveBackgroundGaussian([1, 2]), RemoveBackgroundGaussian([(1, 2, 2), (1, 2, 2)]),
            #RemoveBackgroundGaussian([(1, 2, 2, 3), (1, 2, 2, 3)]),
        ])

    def test_individual_transforms_incorrect_initialisation(self):

        # TODO the commented-out ones do not raise exception during initialisation, but could (to catch errors early)

        with self.assertRaises(BaseException):
            tr = Compose([Resize((30, 40))])

        # with self.assertRaises(BaseException):
        #     tr = Compose([Resize((30, 40, 20, 20))])

        # with self.assertRaises(BaseException):
        #     tr = Compose([Scale((30, 40, 20, 20))])

        # with self.assertRaises(BaseException):
        #     tr = Compose([RandomFlip([1, 3])])

        # with self.assertRaises(BaseException):
        #     tr = Compose([CenterCrop((30, 40))])

        # with self.assertRaises(BaseException):
        #     tr = Compose([CenterCrop((30, 40, 20, 20))])

        # with self.assertRaises(BaseException):
        #     tr = Compose([GaussianNoise(0.5, (1, 2))])

        # with self.assertRaises(BaseException):
        #     tr = Compose([PoissonNoise(0.5)])

        with self.assertRaises(BaseException):
            tr = Compose([NormalizeMeanStd(3, (4, 5))])

        # with self.assertRaises(BaseException):
        #     tr = Compose([GaussianBlur(([1], [2], [2]))])

        # with self.assertRaises(BaseException):
        #     tr = Compose([Normalize(2, [3, 4])])


if __name__ == '__main__':
    unittest.main()
