import unittest

import time
import numpy as np

from src.bio_volumentations.augmentations import functional, atleast_kd, get_nonchannel_axes


class TestNormalization(unittest.TestCase):
    def normalize_cycle(self, img, mean, std):
        # def normalize_channel(img, mean, std):
        #     return (img - img.mean()) * (std / img.std()) + mean

        # for-cycle implementation
        for i in range(img.shape[0]):
            img[i] = (img[i] - img[i].mean()) * (std[i] / img[i].std()) + mean[i]
            # img[i] = normalize_channel(img[i], mean[i], std[i])

        return img

    def normalize_vect(self, img, mean, std):
        mean = atleast_kd(mean, img.ndim)
        std = atleast_kd(std, img.ndim)
        img_mean = atleast_kd(img.mean(axis=get_nonchannel_axes(img)), img.ndim)
        img_std = atleast_kd(img.std(axis=get_nonchannel_axes(img)), img.ndim)

        return (img - img_mean) * (std / img_std) + mean

    def test_normalize_cycle_imlp(self):
        img = np.random.random(size=(3, 5, 6, 7))
        mean = [0, 0, 1]
        std = [1, 2, 1]

        img = self.normalize_cycle(img, mean, std)

        # verify it
        for i in range(img.shape[0]):
            self.assertAlmostEqual(img[i].mean(), mean[i])
            self.assertAlmostEqual(img[i].std(), std[i])

    def test_normalize_vect_imlp(self):
        img = np.random.random(size=(3, 5, 6, 7))
        mean = [0, 0, 1]
        std = [1, 2, 1]

        # numpy-vectorised implementation

        img = self.normalize_vect(img, mean, std)

        # verify it
        for i in range(img.shape[0]):
            self.assertAlmostEqual(img[i].mean(), mean[i])
            self.assertAlmostEqual(img[i].std(), std[i])

    def test_runtime(self):
        n = 30
        img_size = (3, 256, 256, 256)
        img_size = (3, 130, 130, 100, 4)
        # img_size = (8, 200, 200, 200)

        time_cycle = self.measure_exec_time(self.normalize_cycle, n=n, img_size=img_size)
        print(f'Runtime (for cycle implementation): {time_cycle}')

        time_vect = self.measure_exec_time(self.normalize_vect, n=n, img_size=img_size)
        print(f'Runtime (vectorised implementation): {time_vect}')

    def measure_exec_time(self, fn, n=100, img_size=(3, 256, 256, 256)):
        total_time = 0
        for i in range(n):
            img = np.random.random(size=img_size)
            mean = [0, 0, 1] + [-1] * (img_size[0] - 3)
            std = [1, 2, 1] + [3] * (img_size[0] - 3)
            start = time.time()
            res = fn(img, mean, std)
            res2 = res.shape
            end = time.time()
            total_time += (end - start)
        return total_time / n

    def test_normalize_fn_1(self):
        img = np.random.random(size=(3, 5, 6, 7))
        mean = [0, 0, 1]
        std = [1, 2, 1]

        res = functional.normalize(img, mean, std)

        for i in range(res.shape[0]):
            self.assertAlmostEqual(res[i].mean(), mean[i])
            self.assertAlmostEqual(res[i].std(), std[i])

    def test_normalize_fn_2(self):
        img = np.random.random(size=(3, 5, 6, 7, 4))
        mean = [0, 0, 1]
        std = [1, 2, 2]

        res = functional.normalize(img, mean, std)

        for i in range(res.shape[0]):
            self.assertAlmostEqual(res[i].mean(), mean[i])
            self.assertAlmostEqual(res[i].std(), std[i])

    def test_normalize_fn_3(self):
        img = np.random.random(size=(1, 5, 6, 7))
        mean = [1]
        std = [2]

        res = functional.normalize(img, mean, std)

        for i in range(res.shape[0]):
            self.assertAlmostEqual(res[i].mean(), mean[i])
            self.assertAlmostEqual(res[i].std(), std[i])


class TestGaussianBlur(unittest.TestCase):
    def gaussian_blur_vect(self, img, sigma, border_mode, cval):
        from skimage.filters import gaussian
        # If None, input is filtered along all axes. Otherwise, input is filtered along the specified axes.
        # When axes is specified, any tuples used for sigma, order, mode and/or radius must match the length of axes.
        # The ith entry in any of these tuples corresponds to the ith entry in axes.
        # return functional.gaussian_filter(img, sigma=sigma, mode=border_mode, cval=cval, axes=[0])  # compute
        return gaussian(img, sigma=sigma, channel_axis=0, preserve_range=True)  # compute

    def test_gaussian_blur_fn_1(self):
        img = np.random.random(size=(3, 100, 101, 120))
        sigma = [2, 1, 1.5]

        res = self.gaussian_blur_vect(img, sigma, 'reflect', 0)
        res1 = functional.gaussian_blur_stack(img, sigma, 'reflect', 0)

        self.assertTrue(np.allclose(res, res1))


class TestCropPadEtc(unittest.TestCase):
    def test_keypoints_crop_1(self):
        keypoints = [(1, 2, 3), (2, 2, 2), (10, 2, 1), (20, 23, 20), (4, 5, 0)]
        out_shape = np.asarray((5, 5, 5))
        corner_position = np.asarray((1, 0, 1))
        pad = np.asarray((0, 0, 0))

        # res = self.crop_keypoints_cycle(keypoints, corner_position, pad, False, out_shape)
        res = self.crop_keypoints_vect(keypoints, corner_position, pad, False, out_shape)
        self.assertEqual(len(res), 2)
        self.assertTupleEqual(tuple(res[0]), (0, 2, 2))
        self.assertTupleEqual(tuple(res[1]), (1, 2, 1))

        # res = self.crop_keypoints_cycle(keypoints, corner_position, pad, True, out_shape)
        res = self.crop_keypoints_vect(keypoints, corner_position, pad, True, out_shape)
        self.assertEqual(len(res), 5)
        self.assertTupleEqual(tuple(res[0]), (0, 2, 2))
        self.assertTupleEqual(tuple(res[1]), (1, 2, 1))
        self.assertTupleEqual(tuple(res[2]), (9, 2, 0))
        self.assertTupleEqual(tuple(res[3]), (19, 23, 19))
        self.assertTupleEqual(tuple(res[4]), (3, 5, -1))

    def test_keypoints_crop_2(self):
        keypoints = [(1, 2, 3, 0), (2, 2, 2, 1), (10, 2, 1, 3), (20, 23, 20, 1), (4, 5, 0, 0)]
        out_shape = np.asarray((5, 5, 5))
        corner_position = np.asarray((1, 0, 1))
        pad = np.asarray((0, 0, 0))

        # res = self.crop_keypoints_cycle(keypoints, corner_position, pad, False, out_shape)
        res = self.crop_keypoints_vect(keypoints, corner_position, pad, False, out_shape)
        self.assertEqual(len(res), 2)
        self.assertTupleEqual(tuple(res[0]), (0, 2, 2))
        self.assertTupleEqual(tuple(res[1]), (1, 2, 1))

        # res = self.crop_keypoints_cycle(keypoints, corner_position, pad, True, out_shape)
        res = self.crop_keypoints_vect(keypoints, corner_position, pad, True, out_shape)
        self.assertEqual(len(res), 5)
        self.assertTupleEqual(tuple(res[0]), (0, 2, 2))
        self.assertTupleEqual(tuple(res[1]), (1, 2, 1))
        self.assertTupleEqual(tuple(res[2]), (9, 2, 0))
        self.assertTupleEqual(tuple(res[3]), (19, 23, 19))
        self.assertTupleEqual(tuple(res[4]), (3, 5, -1))

    def crop_keypoints_cycle(self, keypoints, crop_position, pad, keep_all, crop_shape):
        res = []
        for keypoint in keypoints:  # keypoints = list of tuples of floats
            k = keypoint[:3] - crop_position + pad  # ignore time-dim keypoint position
            if keep_all or (np.all(k >= 0) and np.all((k + .5) < crop_shape)):
                res.append(k)

        return res

    def crop_keypoints_vect(self, keypoints, crop_position, pad, keep_all, crop_shape):
        keys = np.asarray(keypoints)[:, :3] - crop_position + pad
        if keep_all:
            return keys
        mask = (keys >= 0) & (keys+.5 < crop_shape)
        res = keys[np.sum(mask, axis=1) == 3, :]
        return res

    def pad_keypoints_cycle(self, keypoints, pad):
        res = []
        for coo in keypoints:
            padding = np.array(pad) if len(coo) == 3 else np.array(pad + (0,))
            res.append(coo + padding)
        return res

    def pad_keypoints_vect(self, keypoints, pad):
        keys = np.asarray(keypoints)
        padding = np.asarray(pad if keys.shape[1] == 3 else pad+(0,))  # we only need the 'before' pad size
        return keys + padding

    def test_keypoints_pad_1(self):
        keypoints = [(1, 2, 3), (2, 2, 2), (10, 2, 1), (20, 23, 20), (4, 5, 0)]
        pad = (0, 1, 3)

        # res = self.pad_keypoints_cycle(keypoints, pad)
        res = self.pad_keypoints_vect(keypoints, pad)
        self.assertEqual(len(res), 5)
        self.assertTupleEqual(tuple(res[0]), (1, 3, 6))
        self.assertTupleEqual(tuple(res[1]), (2, 3, 5))
        self.assertTupleEqual(tuple(res[2]), (10, 3, 4))
        self.assertTupleEqual(tuple(res[3]), (20, 24, 23))
        self.assertTupleEqual(tuple(res[4]), (4, 6, 3))

    def test_keypoints_pad_2(self):
        keypoints = [(1, 2, 3, 0), (2, 2, 2, 1), (10, 2, 1, 4), (20, 23, 20, 2), (4, 5, 0, 1)]
        pad = (0, 1, 3)

        # res = self.pad_keypoints_cycle(keypoints, pad)
        res = self.pad_keypoints_vect(keypoints, pad)
        self.assertEqual(len(res), 5)
        self.assertTupleEqual(tuple(res[0]), (1, 3, 6, 0))
        self.assertTupleEqual(tuple(res[1]), (2, 3, 5, 1))
        self.assertTupleEqual(tuple(res[2]), (10, 3, 4, 4))
        self.assertTupleEqual(tuple(res[3]), (20, 24, 23, 2))
        self.assertTupleEqual(tuple(res[4]), (4, 6, 3, 1))

    def transpose_keypoints_cycle(self, keypoints, ax1, ax2):
        res = []
        for k in keypoints:
            k = list(k)
            k[ax1 - 1], k[ax2 - 1] = k[ax2 - 1], k[ax1 - 1]
            res.append(tuple(k))
        return res

    def transpose_keypoints_vect(self, keypoints, ax1, ax2):
        keys = np.asarray(keypoints)
        ax1, ax2 = ax1-1, ax2-1
        keys[:, ax1], keys[:, ax2] = keys[:, ax2], keys[:, ax1].copy()
        return keys

    def test_keypoints_transpose_1(self):
        keypoints = [(1, 2, 3), (2, 2, 2), (10, 2, 1), (20, 23, 20), (4, 5, 0)]

        # res = self.transpose_keypoints_cycle(keypoints, 1, 2)
        res = self.transpose_keypoints_vect(keypoints, 1, 2)
        self.assertEqual(len(res), 5)
        self.assertTupleEqual(tuple(res[0]), (2, 1, 3))
        self.assertTupleEqual(tuple(res[4]), (5, 4, 0))

        # res = self.transpose_keypoints_cycle(keypoints, 1, 3)
        res = self.transpose_keypoints_vect(keypoints, 1, 3)
        self.assertEqual(len(res), 5)
        self.assertTupleEqual(tuple(res[0]), (3, 2, 1))
        self.assertTupleEqual(tuple(res[4]), (0, 5, 4))

        # res = self.transpose_keypoints_cycle(keypoints, 3, 2)
        res = self.transpose_keypoints_vect(keypoints, 3, 2)
        self.assertEqual(len(res), 5)
        self.assertTupleEqual(tuple(res[0]), (1, 3, 2))
        self.assertTupleEqual(tuple(res[4]), (4, 0, 5))

    def test_keypoints_transpose_2(self):
        keypoints = [(1, 2, 3, 1), (2, 2, 2, 1), (10, 2, 1, 1), (20, 23, 20, 1), (4, 5, 0, 1)]

        # res = self.transpose_keypoints_cycle(keypoints, 1, 2)
        res = self.transpose_keypoints_vect(keypoints, 1, 2)
        self.assertEqual(len(res), 5)
        self.assertTupleEqual(tuple(res[0]), (2, 1, 3, 1))
        self.assertTupleEqual(tuple(res[4]), (5, 4, 0, 1))

        # res = self.transpose_keypoints_cycle(keypoints, 1, 3)
        res = self.transpose_keypoints_vect(keypoints, 1, 3)
        self.assertEqual(len(res), 5)
        self.assertTupleEqual(tuple(res[0]), (3, 2, 1, 1))
        self.assertTupleEqual(tuple(res[4]), (0, 5, 4, 1))

        # res = self.transpose_keypoints_cycle(keypoints, 3, 2)
        res = self.transpose_keypoints_vect(keypoints, 3, 2)
        self.assertEqual(len(res), 5)
        self.assertTupleEqual(tuple(res[0]), (1, 3, 2, 1))
        self.assertTupleEqual(tuple(res[4]), (4, 0, 5, 1))

    def flip_keypoints_cycle(self, keypoints, axes, img_shape):
        mult, add = np.ones(3, int), np.zeros(3, int)
        for ax in axes:
            mult[ax - 1] = -1
            add[ax - 1] = img_shape[ax - 1] - 1

        res = []
        for k in keypoints:
            flipped = list(np.array(k[:3]) * mult + add)
            if len(k) == 4:
                flipped.append(k[-1])
            res.append(tuple(flipped))
        return res

    def flip_keypoints_vect(self, keypoints, axes, img_shape):
        keys = np.asarray(keypoints)

        ndim = keys.shape[1]
        mult, add = np.ones(ndim, int), np.zeros(ndim, int)
        for ax in axes:
            mult[ax - 1] = -1
            add[ax - 1] = img_shape[ax - 1] - 1

        flipped = keys * mult + add
        return flipped

    def test_keypoints_flip_1(self):
        keypoints = [(1, 2, 3), (2, 2, 2), (10, 2, 1), (20, 23, 20), (4, 5, 0)]
        img_shape = (25, 25, 25)

        axes = []
        # res = self.flip_keypoints_cycle(keypoints, axes, img_shape)
        res = self.flip_keypoints_vect(keypoints, axes, img_shape)
        self.assertEqual(len(res), 5)
        self.assertTupleEqual(tuple(res[0]), (1, 2, 3))
        self.assertTupleEqual(tuple(res[4]), (4, 5, 0))

        axes = [1]
        # res = self.flip_keypoints_cycle(keypoints, axes, img_shape)
        res = self.flip_keypoints_vect(keypoints, axes, img_shape)
        self.assertEqual(len(res), 5)
        self.assertTupleEqual(tuple(res[0]), (23, 2, 3))
        self.assertTupleEqual(tuple(res[4]), (20, 5, 0))

        axes = [1, 3]
        # res = self.flip_keypoints_cycle(keypoints, axes, img_shape)
        res = self.flip_keypoints_vect(keypoints, axes, img_shape)
        self.assertEqual(len(res), 5)
        self.assertTupleEqual(tuple(res[0]), (23, 2, 21))
        self.assertTupleEqual(tuple(res[4]), (20, 5, 24))

    def test_keypoints_flip_2(self):
        keypoints = [(1, 2, 3, 0), (2, 2, 2, 1), (10, 2, 1, 3), (20, 23, 20, 1), (4, 5, 0, 1)]
        img_shape = (25, 25, 25)

        axes = []
        # res = self.flip_keypoints_cycle(keypoints, axes, img_shape)
        res = self.flip_keypoints_vect(keypoints, axes, img_shape)
        self.assertEqual(len(res), 5)
        self.assertTupleEqual(tuple(res[0]), (1, 2, 3, 0))
        self.assertTupleEqual(tuple(res[4]), (4, 5, 0, 1))

        axes = [1]
        # res = self.flip_keypoints_cycle(keypoints, axes, img_shape)
        res = self.flip_keypoints_vect(keypoints, axes, img_shape)
        self.assertEqual(len(res), 5)
        self.assertTupleEqual(tuple(res[0]), (23, 2, 3, 0))
        self.assertTupleEqual(tuple(res[4]), (20, 5, 0, 1))

        axes = [1, 3]
        # res = self.flip_keypoints_cycle(keypoints, axes, img_shape)
        res = self.flip_keypoints_vect(keypoints, axes, img_shape)
        self.assertEqual(len(res), 5)
        self.assertTupleEqual(tuple(res[0]), (23, 2, 21, 0))
        self.assertTupleEqual(tuple(res[4]), (20, 5, 24, 1))


if __name__ == '__main__':
    unittest.main()
