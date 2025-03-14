import unittest

import numpy as np

from src.bio_volumentations import Compose, RandomAffineTransform, GaussianBlur, Flip


raf_transform = RandomAffineTransform(angle_limit=(20, 20, 20,), translation_limit=(0, 0, 0),
                                      scaling_limit=(0.93, 1.07), border_mode='constant', ival=0, mval=0,
                                      always_apply=True)
gb_transform = GaussianBlur(always_apply=True)
f_transform = Flip(always_apply=True)


def get_int_image(mask=False):
    if not mask:
        return np.random.randint(0, 255, [3, 100, 100, 50], dtype=np.uint8)  # RGB image 100x100x50 vox.
    return np.random.randint(0, 255, [100, 100, 50], dtype=np.uint8)  # mask 100x100x50 vox.


class DataKeywordTests(unittest.TestCase):
    def usecase1(self, img_key, gt_key):
        image_dict_keys = [img_key]
        mask_dict_keys = [gt_key]

        volu_debug_compo = Compose([raf_transform], img_keywords=image_dict_keys, mask_keywords=mask_dict_keys)

        img = np.zeros([3, 100, 100, 50], dtype=np.uint8)  # RGB image 100x100x50 vox.
        gt = np.zeros([100, 100, 50], dtype=bool)  # bool mask

        dl_pipeline_item = {
            img_key: img,
            gt_key: gt
        }

        transformed_res = volu_debug_compo(**dl_pipeline_item)

        return transformed_res

    def test_the_image_keyword_missing(self):
        self.assertIsInstance(self.usecase1('voxels', 'gt_mask'), dict)

    def test_the_image_keyword_not_missing(self):
        self.assertIsInstance(self.usecase1('image', 'gt_mask'), dict)

    def test_duplicate_keyword(self):
        # TODO: we are not checking if keywords are OK --> this test fails (it is the responsibility of the user)
        self.assertRaises(Exception, self.usecase1, 'image', 'image')

    def test_no_image(self):
        volu_debug_compo = Compose([raf_transform])

        gt = np.zeros([100, 100, 50], dtype=bool)  # bool mask

        dl_pipeline_item = {
            'mask': gt
        }

        self.assertRaises(Exception, volu_debug_compo, **dl_pipeline_item)

    def test_no_image_keyword(self):
        volu_debug_compo = Compose([raf_transform], img_keywords=())

        gt = np.zeros([100, 100, 50], dtype=bool)  # bool mask

        dl_pipeline_item = {
            'mask': gt
        }

        self.assertRaises(Exception, volu_debug_compo, **dl_pipeline_item)


class TransformBaseClassesTests(unittest.TestCase):
    def test_IOT_one_image(self):
        volu_debug_compo = Compose([gb_transform], img_keywords=('image',))

        img = get_int_image()

        dl_pipeline_item = {
            'image': img,
        }

        transformed_res = volu_debug_compo(**dl_pipeline_item)

        self.assertIsInstance(transformed_res, dict)
        self.assertEqual(len(transformed_res.keys()), 1)
        self.assertFalse(np.array_equal(transformed_res['image'], img))

    def test_IOT_two_images(self):
        volu_debug_compo = Compose([gb_transform], img_keywords=('image', 'image2'))

        img = get_int_image()
        img2 = get_int_image()

        dl_pipeline_item = {
            'image': img,
            'image2': img2,
        }

        transformed_res = volu_debug_compo(**dl_pipeline_item)

        self.assertIsInstance(transformed_res, dict)
        self.assertEqual(len(transformed_res.keys()), 2)
        self.assertFalse(np.array_equal(transformed_res['image'], img))
        self.assertFalse(np.array_equal(transformed_res['image2'], img2))

    def test_IOT_one_image_annotated(self):
        volu_debug_compo = Compose([gb_transform], img_keywords=('image',))

        img = get_int_image()
        gt = get_int_image(mask=True)

        dl_pipeline_item = {
            'image': img,
            'mask': gt,
        }

        transformed_res = volu_debug_compo(**dl_pipeline_item)

        self.assertIsInstance(transformed_res, dict)
        self.assertEqual(len(transformed_res.keys()), 2)
        self.assertFalse(np.array_equal(transformed_res['image'], img))
        self.assertTrue(np.array_equal(transformed_res['mask'], gt))

    def test_IOT_two_images_annotated(self):
        volu_debug_compo = Compose([gb_transform], img_keywords=('image', 'image2'))

        img = get_int_image()
        img2 = get_int_image()
        gt = get_int_image(mask=True)

        dl_pipeline_item = {
            'image': img,
            'image2': img2,
            'mask': gt,
        }

        transformed_res = volu_debug_compo(**dl_pipeline_item)

        self.assertIsInstance(transformed_res, dict)
        self.assertEqual(len(transformed_res.keys()), 3)
        self.assertFalse(np.array_equal(transformed_res['image'], img))
        self.assertFalse(np.array_equal(transformed_res['image2'], img2))
        self.assertTrue(np.array_equal(transformed_res['mask'], gt))

    def test_DT_one_image(self):
        volu_debug_compo = Compose([f_transform], img_keywords=('image',))

        img = get_int_image()

        dl_pipeline_item = {
            'image': img,
        }

        transformed_res = volu_debug_compo(**dl_pipeline_item)

        self.assertIsInstance(transformed_res, dict)
        self.assertEqual(len(transformed_res.keys()), 1)
        self.assertFalse(np.array_equal(transformed_res['image'], img))

    def test_DT_two_images(self):
        volu_debug_compo = Compose([f_transform], img_keywords=('image', 'image2'))

        img = get_int_image()
        img2 = get_int_image()

        dl_pipeline_item = {
            'image': img,
            'image2': img2,
        }

        transformed_res = volu_debug_compo(**dl_pipeline_item)

        self.assertIsInstance(transformed_res, dict)
        self.assertEqual(len(transformed_res.keys()), 2)
        self.assertFalse(np.array_equal(transformed_res['image'], img))
        self.assertFalse(np.array_equal(transformed_res['image2'], img2))

    def test_DT_one_image_annotated(self):
        volu_debug_compo = Compose([f_transform], img_keywords=('image',))

        img = get_int_image()
        gt = get_int_image(mask=True)

        dl_pipeline_item = {
            'image': img,
            'mask': gt,
        }

        transformed_res = volu_debug_compo(**dl_pipeline_item)

        self.assertIsInstance(transformed_res, dict)
        self.assertEqual(len(transformed_res.keys()), 2)
        self.assertFalse(np.array_equal(transformed_res['image'], img))
        self.assertFalse(np.array_equal(transformed_res['mask'], gt))

    def test_DT_two_images_annotated(self):
        volu_debug_compo = Compose([f_transform], img_keywords=('image', 'image2'))

        img = get_int_image()
        img2 = get_int_image()
        gt = get_int_image(mask=True)

        dl_pipeline_item = {
            'image': img,
            'image2': img2,
            'mask': gt,
        }

        transformed_res = volu_debug_compo(**dl_pipeline_item)

        self.assertIsInstance(transformed_res, dict)
        self.assertEqual(len(transformed_res.keys()), 3)
        self.assertFalse(np.array_equal(transformed_res['image'], img))
        self.assertFalse(np.array_equal(transformed_res['image2'], img2))
        self.assertFalse(np.array_equal(transformed_res['mask'], gt))


if __name__ == '__main__':
    unittest.main()
