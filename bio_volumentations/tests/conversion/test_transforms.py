import unittest
import numpy as np
import torch
from volumentations.conversion.transforms import *


class TestNoConversion(unittest.TestCase):
    def test_image(self):
        sh = (36, 200, 250)
        img = np.empty(sh, dtype=float)
        tr = NoConversion()
        result = tr(force_apply=True, targets=[['image']], image=img)
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
        result = tr(force_apply=True, targets=[['image'], ['mask']], image=img, mask=lbl)
        self.assertIsInstance(result, dict)
        res_image = result['image']
        res_label = result['mask']

        self.assertIsInstance(res_image, np.ndarray)
        self.assertTupleEqual(tuple(res_image.shape), sh + (2,))
        self.assertListEqual(res_image.tolist(), img.tolist())
        self.assertIsInstance(res_label, np.ndarray)
        self.assertTupleEqual(tuple(res_label.shape), sh)
        self.assertListEqual(res_label.tolist(), lbl.tolist())


class TestToTensor(unittest.TestCase):
    def test_image3d(self):
        sh = (36, 200, 250)
        img = np.empty(sh)
        tr = ToTensor()
        res_image = tr(force_apply=True, targets=[['image']], image=img)['image']
        self.assertIsInstance(res_image, torch.Tensor)
        self.assertTupleEqual(tuple(res_image.shape), sh)
        self.assertListEqual(res_image.tolist(), img.tolist())

    def test_image4d(self):
        sh = (36, 200, 250)
        img = np.empty(sh + (4,))
        tr = ToTensor()
        res_image = tr(force_apply=True, targets=[['image']], image=img)['image']
        self.assertIsInstance(res_image, torch.Tensor)
        self.assertTupleEqual(tuple(res_image.shape), (4,) + sh)
        self.assertListEqual(res_image.tolist(), img.transpose((3, 0, 1, 2)).tolist())

    def test_image_labels_transpose(self):
        sh = (36, 200, 250)
        img = np.empty(sh + (2,), dtype=float)
        lbl = np.empty(sh, dtype=float)
        tr = ToTensor()
        result = tr(force_apply=True, targets=[['image'], ['mask']], image=img, mask=lbl)
        res_image = result['image']
        res_label = result['mask']
        self.assertIsInstance(res_image, torch.Tensor)
        self.assertTupleEqual(tuple(res_image.shape), (2,) + sh)
        self.assertListEqual(res_image.tolist(), img.transpose((3, 0, 1, 2)).tolist())
        self.assertIsInstance(res_label, torch.Tensor)
        self.assertTupleEqual(tuple(res_label.shape), sh)
        self.assertListEqual(res_label.tolist(), lbl.tolist())

        lbl = np.empty(sh + (3,), dtype=float)
        tr = ToTensor()
        result = tr(force_apply=True, targets=[['image'], ['mask']], image=img, mask=lbl)
        res_image = result['image']
        res_label = result['mask']
        self.assertIsInstance(res_image, torch.Tensor)
        self.assertTupleEqual(tuple(res_image.shape), (2,) + sh)
        self.assertListEqual(res_image.tolist(), img.transpose((3, 0, 1, 2)).tolist())
        self.assertIsInstance(res_label, torch.Tensor)
        self.assertTupleEqual(tuple(res_label.shape), (3,) + sh)
        self.assertListEqual(res_label.tolist(), lbl.transpose((3, 0, 1, 2)).tolist())

    def test_image_labels_no_transpose(self):
        sh = (36, 200, 250)
        img = np.empty(sh + (2,), dtype=float)
        lbl = np.empty(sh + (3,), dtype=float)
        tr = ToTensor(transpose_mask=False)
        result = tr(force_apply=True, targets=[['image'], ['mask']], image=img, mask=lbl)
        res_image = result['image']
        res_label = result['mask']
        self.assertIsInstance(res_image, torch.Tensor)
        self.assertTupleEqual(tuple(res_image.shape), (2,) + sh)
        self.assertListEqual(res_image.tolist(), img.transpose((3, 0, 1, 2)).tolist())
        self.assertIsInstance(res_label, torch.Tensor)
        self.assertTupleEqual(tuple(res_label.shape), sh + (3,))
        self.assertListEqual(res_label.tolist(), lbl.tolist())


if __name__ == '__main__':
    unittest.main()
