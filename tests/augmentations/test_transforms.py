import unittest
import numpy as np
from bio_volumentations.augmentations.transforms import *
from bio_volumentations.core.composition import Compose


class TestResize(unittest.TestCase):
    def test_1(self):
        tr = Compose([Resize(shape=(20, 20, 20))])

        img = np.empty((30, 30, 30))
        tr_img = tr(image=img)['image']
        self.assertTupleEqual(tr_img.shape, (1, 20, 20, 20))

        img = np.empty((1, 30, 30, 30))
        tr_img = tr(image=img)['image']
        self.assertTupleEqual(tr_img.shape, (1, 20, 20, 20))

        img = np.empty((4, 30, 30, 30))
        tr_img = tr(image=img)['image']
        self.assertTupleEqual(tr_img.shape, (4, 20, 20, 20))

        img = np.empty((4, 30, 30, 30, 5))
        tr_img = tr(image=img)['image']
        self.assertTupleEqual(tr_img.shape, (4, 20, 20, 20, 5))

    def test_2(self):
        tr = Compose([Resize(shape=(20, 20, 20, 6))])

        # when arguments are incorrect, should print warning and output the unchanged input image
        img = np.empty((30, 30, 30))
        tr_img = tr(image=img)['image']
        self.assertTupleEqual(tr_img.shape, (1, 30, 30, 30))

        img = np.empty((1, 30, 30, 30))
        tr_img = tr(image=img)['image']
        self.assertTupleEqual(tr_img.shape, (1, 30, 30, 30))

        img = np.empty((4, 30, 30, 30))
        tr_img = tr(image=img)['image']
        self.assertTupleEqual(tr_img.shape, (4, 30, 30, 30))

        img = np.empty((4, 30, 30, 30, 5))
        tr_img = tr(image=img)['image']
        self.assertTupleEqual(tr_img.shape, (4, 20, 20, 20, 6))


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


if __name__ == '__main__':
    unittest.main()
