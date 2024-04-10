import unittest
import numpy as np
from volumentations.augmentations.transforms import *


class TestIncreaseDimensionality(unittest.TestCase):
    def test_increase(self):
        sh = (36, 200, 250)
        img = np.empty(sh + (3,))
        tr = IncreaseDimensionality(6)
        res_image = tr(force_apply=True, targets=[['image']], image=img)['image']
        self.assertIsInstance(res_image, np.ndarray)
        self.assertTupleEqual(tuple(res_image.shape), sh + (3, 1, 1))

        lbl = np.empty(sh)
        result = tr(force_apply=True, targets=[['image'], ['mask']], image=img, mask=lbl)
        res_image = result['image']
        res_label = result['mask']
        self.assertIsInstance(res_image, np.ndarray)
        self.assertTupleEqual(tuple(res_image.shape), sh + (3, 1, 1))
        self.assertIsInstance(res_label, np.ndarray)
        self.assertTupleEqual(tuple(res_label.shape), sh + (1, 1))

    def test_decrease(self):
        sh = (36, 200, 250, 4)
        img = np.empty(sh)
        tr = IncreaseDimensionality(3)
        res_image = tr(force_apply=True, targets=[['image']], image=img)['image']
        self.assertIsInstance(res_image, np.ndarray)
        self.assertTupleEqual(tuple(res_image.shape), sh)
        self.assertListEqual(res_image.tolist(), img.tolist())

        sh = (36, 200, 250)
        img = np.empty(sh)
        tr = IncreaseDimensionality(1)
        res_image = tr(force_apply=True, targets=[['image']], image=img)['image']
        self.assertIsInstance(res_image, np.ndarray)
        self.assertTupleEqual(tuple(res_image.shape), sh)
        self.assertListEqual(res_image.tolist(), img.tolist())

    def test_no_change(self):
        sh = (36, 200, 250)
        img = np.empty(sh)
        tr = IncreaseDimensionality(3)
        res_image = tr(force_apply=True, targets=[['image']], image=img)['image']
        self.assertIsInstance(res_image, np.ndarray)
        self.assertTupleEqual(tuple(res_image.shape), sh)
        self.assertListEqual(res_image.tolist(), img.tolist())

        img = np.empty(sh + (3,))
        lbl = np.empty(sh)
        tr = IncreaseDimensionality(4)
        result = tr(force_apply=True, targets=[['image'], ['mask']], image=img, mask=lbl)
        res_image = result['image']
        res_label = result['mask']
        self.assertIsInstance(res_image, np.ndarray)
        self.assertTupleEqual(tuple(res_image.shape), sh + (3,))
        self.assertIsInstance(res_label, np.ndarray)
        self.assertTupleEqual(tuple(res_label.shape), sh)

    def test_negative(self):
        sh = (36, 200, 250)
        img = np.empty(sh)
        tr = IncreaseDimensionality(-1)
        result = tr(force_apply=True, targets=[['image']], image=img)
        res_image = result['image']
        self.assertIsInstance(res_image, np.ndarray)
        self.assertTupleEqual(tuple(res_image.shape), sh)
        self.assertListEqual(res_image.tolist(), img.tolist())


class TestNormalizeMeanStd(unittest.TestCase):
    def test_random_data3d(self):
        sh = (36, 200, 250)
        img = np.random.rand(*sh)
        tr = NormalizeMeanStd(mean=[img.mean()], std=[img.std()], max_pixel_value=1)
        res_image = tr(force_apply=True, targets=[['image']], image=img)['image']
        self.assertFalse(np.all(np.equal(img, res_image)))

    def test_random_data4d(self):
        sh = (36, 200, 250, 4)
        img = np.random.rand(*sh)
        tr = NormalizeMeanStd(mean=[img.mean(axis=(0, 1, 2))], std=[img.std(axis=(0, 1, 2))], max_pixel_value=1)
        res_image = tr(force_apply=True, targets=[['image']], image=img)['image']
        self.assertFalse(np.all(np.equal(img, res_image)))

    def test_constant_data(self):
        sh = (36, 200, 250)
        img = np.ones(sh)
        tr = NormalizeMeanStd(mean=[1], std=[0], max_pixel_value=1)
        res_image = tr(force_apply=True, targets=[['image']], image=img)['image']
        self.assertTrue(np.all(np.equal(res_image, np.zeros_like(img))))

        sh = (36, 200, 250)
        img = np.ones(sh + (2,))
        tr = NormalizeMeanStd(mean=[1, 0.5], std=[0, 0], max_pixel_value=1)
        res_image = tr(force_apply=True, targets=[['image']], image=img)['image']
        self.assertTrue(np.all(np.equal(res_image[..., 0], np.zeros(sh))))
        self.assertTrue(np.all(np.equal(res_image[..., 1], np.zeros(sh) + 0.5)))

    def test_random_data4d_labels(self):
        sh = (36, 200, 250, 4)
        img = np.random.rand(*sh)
        lbl = np.random.rand(*sh[:3])
        tr = NormalizeMeanStd(mean=[img.mean(axis=(0, 1, 2))], std=[img.std(axis=(0, 1, 2))], max_pixel_value=1)
        result = tr(force_apply=True, targets=[['image'], ['mask']], image=img, mask=lbl)
        res_image = result['image']
        res_label = result['mask']
        self.assertFalse(np.all(np.equal(img, res_image)))
        self.assertTrue(np.all(np.equal(lbl, res_label)))


if __name__ == '__main__':
    unittest.main()
