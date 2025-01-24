import unittest
import numpy as np
from src.conversion.transforms import *


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
        # result = tr(force_apply=True, targets=[['image'], ['mask']], image=img, mask=lbl)
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


if __name__ == '__main__':
    unittest.main()
