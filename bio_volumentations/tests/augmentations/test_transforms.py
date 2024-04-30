# ============================================================================================= #
#  Author:       Filip Lux                                                                      #
#  Copyright:    Filip Lux          : lux.filip@gmail.com                                       #
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
    NormalizeMeanStd, GaussianBlur, Normalize, HistogramEqualization, RandomBrightnessContrast)
from bio_volumentations.core.composition import Compose
import numpy as np


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


class TestRandomScale(unittest.TestCase):
    def test_shape(self):

        limits = [0.2,
                  (0.8, 1.2),
                  (0.2, 0.3, 0.1),
                  (0.8, 1.2, 0.9, 1.1, 0.7, 1.)]

        for scaling_limit in limits:
            tests = get_shape_tests(RandomScale, (31, 32, 33),
                                    params={'scaling_limit': scaling_limit})
            for tr_img, expected_shape, data_type in tests:
                self.assertTupleEqual(tr_img.shape, expected_shape)
                self.assertEqual(tr_img.dtype, data_type)


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


class TestFlip(unittest.TestCase):
    def test_shape(self):
        tests = get_shape_tests(Flip, (31, 32, 33))
        for tr_img, expected_shape, data_type in tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)


class TestRandomFlip(unittest.TestCase):
    def test_shape(self):

        axes_list = [None,
                     [],
                     [1],
                     [1, 2],
                     [1, 2, 3]]

        for axes in axes_list:
            tests = get_shape_tests(RandomFlip, (30, 30, 30),
                                    params={'axes_to_choose': axes})
            for tr_img, expected_shape, data_type in tests:
                self.assertTupleEqual(tr_img.shape, expected_shape)
                self.assertEqual(tr_img.dtype, data_type)


class TestCenterCrop(unittest.TestCase):
    def test_inflate(self):
        in_shape = (32, 31, 30)
        shape_tests = get_shape_tests(CenterCrop, in_shape, {'shape': (40, 41, 42)})

        for tr_img, expected_shape, data_type in shape_tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)

    def test_deflate(self):
        in_shape = (32, 31, 30)
        shape_tests = get_shape_tests(CenterCrop, in_shape, {'shape': (20, 21, 22)})

        for tr_img, expected_shape, data_type in shape_tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)
            self.assertEqual(tr_img.dtype, data_type)


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


def get_shape_tests(transform, in_shape: tuple, params={}):
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


class TestGaussianBlur(unittest.TestCase):
    def test_shape(self):
        tests = get_shape_tests(GaussianBlur, (31, 32, 33))
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


if __name__ == '__main__':
    unittest.main()
