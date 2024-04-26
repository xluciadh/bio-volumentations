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
        for output, expected_shape in tests:
            self.assertTupleEqual(output.shape, expected_shape)


class TestPoissonNoise(unittest.TestCase):
    def test_shape(self):
        tests = get_shape_tests(PoissonNoise, (31, 32, 33))
        for output, expected_shape in tests:
            self.assertTupleEqual(output.shape, expected_shape)


class TestScale(unittest.TestCase):
    def test_shape(self):
        tests = get_shape_tests(Scale, (31, 32, 33), params={'scales': 1.5})
        for output, expected_shape in tests:
            self.assertTupleEqual(output.shape, expected_shape)

        tests = get_shape_tests(Scale, (31, 32, 33), params={'scales': 0.8})
        for output, expected_shape in tests:
            self.assertTupleEqual(output.shape, expected_shape)


class TestRandomScale(unittest.TestCase):
    def test_shape(self):

        limits = [0.2,
                  (0.8, 1.2),
                  (0.2, 0.3, 0.1),
                  (0.8, 1.2, 0.9, 1.1, 0.7, 1.)]

        for scaling_limit in limits:
            tests = get_shape_tests(RandomScale, (31, 32, 33),
                                    params={'scaling_limit': scaling_limit})
            for output, expected_shape in tests:
                self.assertTupleEqual(output.shape, expected_shape)


class TestRandomRotate90(unittest.TestCase):
    def test_shape(self):

        axes_list = [None,
                     [1],
                     [1, 2],
                     [1, 2, 3]]

        for axes in axes_list:
            tests = get_shape_tests(RandomRotate90, (30, 30, 30),
                                    params={'axes': axes})
            for output, expected_shape in tests:
                self.assertTupleEqual(output.shape, expected_shape)


class TestFlip(unittest.TestCase):
    def test_shape(self):
        tests = get_shape_tests(Flip, (31, 32, 33))
        for output, expected_shape in tests:
            self.assertTupleEqual(output.shape, expected_shape)


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
            for output, expected_shape in tests:
                self.assertTupleEqual(output.shape, expected_shape)


class TestCenterCrop(unittest.TestCase):
    def test_inflate(self):
        in_shape = (32, 31, 30)
        shape_tests = get_shape_tests(CenterCrop, in_shape, {'shape': (40, 41, 42)})

        for tr_img, expected_shape in shape_tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)

    def test_deflate(self):
        in_shape = (32, 31, 30)
        shape_tests = get_shape_tests(CenterCrop, in_shape, {'shape': (20, 21, 22)})

        for tr_img, expected_shape in shape_tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)


class TestRandomCrop(unittest.TestCase):
    def test_inflate(self):
        in_shape = (32, 31, 30)
        shape_tests = get_shape_tests(RandomCrop, in_shape, {'shape': (40, 41, 42)})

        for tr_img, expected_shape in shape_tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)

    def test_deflate(self):
        in_shape = (32, 31, 30)
        shape_tests = get_shape_tests(RandomCrop, in_shape, {'shape': (20, 21, 22)})

        for tr_img, expected_shape in shape_tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)


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
    img = np.ones((w, h, d))
    mask = np.ones((w, h, d))
    tr_img = tr(image=img, mask=mask)
    res.append((tr_img['image'], (1, w_, h_, d_)))
    res.append((tr_img['mask'], (w_, h_, d_)))

    # img (C, W, H, D), mask (W, H, D)
    img = np.ones((4, w, h, d))
    mask = np.ones((w, h, d))
    tr_img = tr(image=img, mask=mask)
    res.append((tr_img['image'], (4, w_, h_, d_)))
    res.append((tr_img['mask'], (w_, h_, d_)))

    # img (C, W, H, D, T), mask (W, H, D, T)
    img = np.ones((4, w, h, d, 5))
    mask = np.ones((w, h, d, 5))
    tr_img = tr(image=img, mask=mask)
    res.append((tr_img['image'], (4, w_, h_, d_, 5)))
    res.append((tr_img['mask'], (w_, h_, d_, 5)))

    return res


class TestResize(unittest.TestCase):
    def test_inflate(self):
        in_shape = (32, 31, 30)
        shape_tests = get_shape_tests(Resize, in_shape, {'shape': (40, 41, 42)})

        for tr_img, expected_shape in shape_tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)

    def test_deflate(self):
        in_shape = (32, 31, 30)
        shape_tests = get_shape_tests(Resize, in_shape, {'shape': (20, 21, 22)})

        for tr_img, expected_shape in shape_tests:
            self.assertTupleEqual(tr_img.shape, expected_shape)


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
            for output, expected_shape in tests:
                self.assertTupleEqual(output.shape, expected_shape)

        scale_limits = [0.2,
                        (0.8, 1.2),
                        (0.2, 0.3, 0.1),
                        (0.8, 1.2, 0.9, 1.1, 0.7, 1.)]

        for scale_limit in scale_limits:
            tests = get_shape_tests(RandomAffineTransform, (31, 32, 33),
                                    params={'scaling_limit': scale_limit})
            for output, expected_shape in tests:
                self.assertTupleEqual(output.shape, expected_shape)

        translation_limits = [10,
                              (0, 12),
                              (3, 5, 10),
                              (-3, 3, -5, 5, 0, 0)]

        for translation in translation_limits:
            tests = get_shape_tests(RandomAffineTransform, (31, 32, 33),
                                    params={'translation_limit': translation})
            for output, expected_shape in tests:
                self.assertTupleEqual(output.shape, expected_shape)


class TestAffineTransform(unittest.TestCase):
    def test_shape(self):

        scale = (1.2, 0.8, 1)
        translation = (0, 1, -40)
        angles = (-20, 0, -0.5)

        tests = get_shape_tests(AffineTransform, (31, 32, 33),
                                params={'translation': translation})
        for output, expected_shape in tests:
            self.assertTupleEqual(output.shape, expected_shape)

        tests = get_shape_tests(AffineTransform, (31, 32, 33),
                                params={'scale': scale})
        for output, expected_shape in tests:
            self.assertTupleEqual(output.shape, expected_shape)

        tests = get_shape_tests(AffineTransform, (31, 32, 33),
                                params={'angles': angles})
        for output, expected_shape in tests:
            self.assertTupleEqual(output.shape, expected_shape)


class TestNormalizeMeanStd(unittest.TestCase):
    def test_shape(self):

        mean = 1.2
        std = 2
        tests = get_shape_tests(NormalizeMeanStd, (31, 32, 33),
                                params={'mean': mean,
                                        'std': std})
        for output, expected_shape in tests:
            self.assertTupleEqual(output.shape, expected_shape)


class TestGaussianBlur(unittest.TestCase):
    def test_shape(self):
        tests = get_shape_tests(GaussianBlur, (31, 32, 33))
        for output, expected_shape in tests:
            self.assertTupleEqual(output.shape, expected_shape)


class TestRandomGamma(unittest.TestCase):
    def test_shape(self):
        tests = get_shape_tests(RandomGamma, (31, 32, 33))
        for output, expected_shape in tests:
            self.assertTupleEqual(output.shape, expected_shape)


class TestRandomBrightnessContrast(unittest.TestCase):
    def test_shape(self):

        brightness_list = [3, (2, 5)]
        contrast_list = [0.5, (.7, 1.1)]


        for brightness in brightness_list:
            for contrast in contrast_list:
                tests = get_shape_tests(RandomBrightnessContrast, (30, 31, 32),
                                        params={'brightness_limit': brightness,
                                                'contrast_limit': contrast})
                for output, expected_shape in tests:
                    self.assertTupleEqual(output.shape, expected_shape)


class TestHistogramEqualization(unittest.TestCase):
    def test_shape(self):
        tests = get_shape_tests(HistogramEqualization, (31, 32, 33))
        for output, expected_shape in tests:
            self.assertTupleEqual(output.shape, expected_shape)


class TestNormalize(unittest.TestCase):
    def test_shape(self):
        tests = get_shape_tests(Normalize, (31, 32, 33))
        for output, expected_shape in tests:
            self.assertTupleEqual(output.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
