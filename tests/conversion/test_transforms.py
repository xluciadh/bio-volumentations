# ============================================================================================= #
#  Author:       Lucia Hradecká                                                                 #
#  Copyright:    Lucia Hradecká     : lucia.d.hradecka@gmail.com                                #
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

from src.bio_volumentations.conversion.transforms import *
from src.bio_volumentations.core.composition import Compose
from src.bio_volumentations.augmentations import transforms as T


class TestNoConversion(unittest.TestCase):
    def test_image(self):
        sh = (36, 200, 250)
        img = np.empty(sh, dtype=float)
        tr = NoConversion()
        result = tr(force_apply=True, targets={'img_keywords': ['image']}, image=img)
        self.assertIsInstance(result, dict)
        res_image = result['image']
        self.assertIsInstance(res_image, np.ndarray)
        self.assertTupleEqual(tuple(res_image.shape), sh)
        self.assertListEqual(res_image.tolist(), img.tolist())

    def test_image_labels(self):
        sh = (36, 200, 250)
        img = np.empty(sh + (2,), dtype=float)
        lbl = np.empty(sh, dtype=float)
        tr = NoConversion()
        result = tr(force_apply=True, targets={'img_keywords': ['image'], 'mask_keywords': ['mask']}, image=img, mask=lbl)
        self.assertIsInstance(result, dict)
        res_image = result['image']
        res_label = result['mask']

        self.assertIsInstance(res_image, np.ndarray)
        self.assertTupleEqual(tuple(res_image.shape), sh + (2,))
        self.assertListEqual(res_image.tolist(), img.tolist())
        self.assertIsInstance(res_label, np.ndarray)
        self.assertTupleEqual(tuple(res_label.shape), sh)
        self.assertListEqual(res_label.tolist(), lbl.tolist())


class TestChecks(unittest.TestCase):
    def test_dim_shape_checks_OK(self):
        sh = (5, 6, 7)
        img1 = np.empty(sh)
        img2 = np.empty(sh)
        mask1 = np.empty(sh)
        float_mask1 = np.empty(sh)
        float_mask2 = np.empty(sh)
        kpts1 = [(1, 1, 1), (2, 2, 2)]
        kpts2 = [(1, 1, 1), (2, 2, 2)]

        tr_pip = Compose([], img_keywords=('image', 'image2'), fmask_keywords=('fm1', 'fm2'), keypoints_keywords=('kpts1', 'kpts2'))
        res_dict = tr_pip(image=img1, image2=img2, mask=mask1, fm1=float_mask1, fm2=float_mask2, kpts1=kpts1, kpts2=kpts2)

    def test_dim_shape_checks_NOK(self):
        img1 = np.empty((5, 6, 7))
        img2 = np.empty((5, 6, 70))
        img3 = np.empty((1, 5, 6, 7))
        img4 = np.empty((1, 5, 6, 7, 8))

        mask1 = np.empty((5, 6, 7))
        float_mask1 = np.empty((5, 6, 7))
        float_mask2 = np.empty((5, 60, 7))
        float_mask3 = np.empty((5, 6, 7, 8))

        kpts1 = [(1, 1, 1), (2, 2, 2)]
        kpts2 = [(1, 1, 1), (2, 2, 2), (1, 2, 3, 4)]
        kpts3 = [(1, 2, 3, 4)]

        # OK
        tr_pip = Compose([], img_keywords=('image', 'image2'), fmask_keywords=('fm1', 'fm2'), keypoints_keywords=('kpts1', 'kpts2'))
        res_dict = tr_pip(image=img1, image2=img3, mask=mask1, fm1=float_mask1, kpts1=kpts1)

        # NOK
        with self.assertRaises((ValueError, RuntimeError)):
            tr_pip = Compose([], img_keywords=('image', 'image2'), fmask_keywords=('fm1', 'fm2'), keypoints_keywords=('kpts1', 'kpts2'))
            res_dict = tr_pip(image=img1, image2=img2)

        with self.assertRaises((ValueError, RuntimeError)):
            tr_pip = Compose([], img_keywords=('image', 'image2'), fmask_keywords=('fm1', 'fm2'), keypoints_keywords=('kpts1', 'kpts2'))
            res_dict = tr_pip(image=img1, image2=img4)

        with self.assertRaises((ValueError, RuntimeError)):
            tr_pip = Compose([], img_keywords=('image', 'image2'), fmask_keywords=('fm1', 'fm2'), keypoints_keywords=('kpts1', 'kpts2'))
            res_dict = tr_pip(image=img3, image2=img4)

        with self.assertRaises((ValueError, RuntimeError)):
            tr_pip = Compose([], img_keywords=('image', 'image2'), fmask_keywords=('fm1', 'fm2'), keypoints_keywords=('kpts1', 'kpts2'))
            res_dict = tr_pip(image=img1, fm1=float_mask2)

        with self.assertRaises((ValueError, RuntimeError)):
            tr_pip = Compose([], img_keywords=('image', 'image2'), fmask_keywords=('fm1', 'fm2'), keypoints_keywords=('kpts1', 'kpts2'))
            res_dict = tr_pip(image=img1, fm1=float_mask3)

        with self.assertRaises((ValueError, RuntimeError)):
            tr_pip = Compose([], img_keywords=('image', 'image2'), fmask_keywords=('fm1', 'fm2'), keypoints_keywords=('kpts1', 'kpts2'))
            res_dict = tr_pip(image=img4, fm1=float_mask1)

        with self.assertRaises((ValueError, RuntimeError)):
            tr_pip = Compose([], img_keywords=('image', 'image2'), fmask_keywords=('fm1', 'fm2'), keypoints_keywords=('kpts1', 'kpts2'))
            res_dict = tr_pip(image=img2, fm1=float_mask2)

        with self.assertRaises((ValueError, RuntimeError)):
            tr_pip = Compose([], img_keywords=('image', 'image2'), fmask_keywords=('fm1', 'fm2'), keypoints_keywords=('kpts1', 'kpts2'))
            res_dict = tr_pip(image=img1, kpts1=kpts2)

        with self.assertRaises((ValueError, RuntimeError)):
            tr_pip = Compose([], img_keywords=('image', 'image2'), fmask_keywords=('fm1', 'fm2'), keypoints_keywords=('kpts1', 'kpts2'))
            res_dict = tr_pip(image=img1, kpts1=kpts3)

        with self.assertRaises((ValueError, RuntimeError)):
            tr_pip = Compose([], img_keywords=('image', 'image2'), fmask_keywords=('fm1', 'fm2'), keypoints_keywords=('kpts1', 'kpts2'))
            res_dict = tr_pip(image=img1, kpts1=kpts1, kpts2=kpts3)

        with self.assertRaises((ValueError, RuntimeError)):
            tr_pip = Compose([], img_keywords=('image', 'image2'), fmask_keywords=('fm1', 'fm2'), keypoints_keywords=('kpts1', 'kpts2'))
            res_dict = tr_pip(image=img4, kpts1=kpts1)

        with self.assertRaises((ValueError, RuntimeError)):
            tr_pip = Compose([], img_keywords=('image', 'image2'), fmask_keywords=('fm1', 'fm2'), keypoints_keywords=('kpts1', 'kpts2'))
            res_dict = tr_pip(fm1=float_mask1, kpts1=kpts1)

    def test_kpts_format_fix(self):
        img = np.empty((5, 6, 7))

        kpts1 = [(1, 1, 1), (2, 2, 2)]
        kpts2 = [[1, 1, 1], (2, 2, 2)]
        kpts3 = [[1, 1, 1], [2, 2, 2]]
        kpts4 = ([1, 1, 1], [2, 2, 2])
        kpts5 = ((1, 1, 1), (2, 2, 2))
        kpts6 = np.asarray(((1, 1, 1), (2, 2, 2)))

        tr_pip = Compose([])
        for k in [kpts1, kpts2, kpts3, kpts4, kpts5, kpts6]:
            res = tr_pip(image=img, keypoints=k)['keypoints']
            self.assertIsInstance(res, list)
            self.assertIsInstance(res[0], tuple)

        tr_pip = Compose([T.AffineTransform(translation=(1, 0, 1)), T.RandomFlip()])
        for k in [kpts1, kpts2, kpts3, kpts4, kpts5, kpts6]:
            res = tr_pip(image=img, keypoints=k)['keypoints']
            self.assertIsInstance(res, list)
            self.assertIsInstance(res[0], tuple)


if __name__ == '__main__':
    unittest.main()
