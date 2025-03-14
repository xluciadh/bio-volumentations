import unittest

import numpy as np

from src.bio_volumentations.core.composition import Compose
import src.bio_volumentations.augmentations.transforms as transforms


class TestComposeConversion(unittest.TestCase):
    def test_no_conversion(self):
        sh = (36, 200, 250, 3)
        img = np.zeros(sh)

        tr = Compose([transforms.NormalizeMeanStd(mean=[0],
                                                  std=[0])])
        result = tr(image=img)

        res_image = result['image']
        self.assertIsInstance(res_image, np.ndarray)

    '''
    def test_pytorch_tensor(self):
        sh = (36, 200, 250, 3)
        img = np.zeros(sh)

        tr = Compose([T.NormalizeMeanStd(mean=[0], std=[0], max_pixel_value=1)], conversion=CT.ToTensor())
        result = tr(image=img)

        res_image = result['image']
        self.assertIsInstance(res_image, torch.Tensor)
        self.assertTrue(np.all(np.equal(img, res_image.numpy().transpose((1, 2, 3, 0)))))
    '''

    def test_compose(self):
        import numpy as np
        from src import Compose, RandomGamma, RandomFlip, GaussianBlur

        # Create the transformation using Compose from a list of transformations and define targets
        aug = Compose([
            RandomGamma(gamma_limit=(0.8, 1.2), p=0.8),
            RandomFlip(axes_to_choose=[1, 2, 3], p=1),
            GaussianBlur(sigma=1.2, p=0.8)
        ],
            img_keywords=('image', 'abc'), mask_keywords=('mask',), fmask_keywords=('nothing',))

        # Generate the image data
        img = np.random.rand(1, 128, 256, 256)
        img1 = np.random.rand(1, 128, 256, 256)
        lbl = np.random.randint(0, 1, size=(128, 256, 256), dtype=np.uint8)

        # Transform the images
        # Notice that the images must be passed as keyword arguments to the transformation pipeline
        # and extracted from the outputted dictionary.
        data = {'image': img, 'abc': img1, 'mask': lbl}
        aug_data = aug(**data)
        transformed_img = aug_data['image']
        transformed_img1 = aug_data['abc']
        transformed_lbl = aug_data['mask']

        self.assertIsInstance(transformed_img, np.ndarray)
        self.assertTupleEqual(img.shape, transformed_img.shape)

    def test_compose2(self):
        from src import Compose, RandomScale, RandomCrop, RandomFlip, RandomBrightnessContrast, RandomGaussianBlur, \
            NormalizeMeanStd

        augmentation_pipeline = Compose([
            RandomScale(scaling_limit=(1.1, 1.6, 0.4, 0.6, 0.4, 0.6), always_apply=True),
            # match the size of nuclei in target data: enlarge in z axis and shrink in x and y axes
            RandomCrop(shape=(32, 320, 320), always_apply=True),  # crop to the desired model input shape
            RandomFlip(axes_to_choose=None, p=1),  # random flipping
            RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, always_apply=True),
            # match intensity characteristics of the target data
            RandomGaussianBlur(max_sigma=(1, 1.5, 1.5), p=0.8),  # match the noise/blur characteristics
            NormalizeMeanStd(mean=35.27, std=27.42, always_apply=True)  # normalize voxel values
        ], img_keywords=('image',), mask_keywords=('mask', 'centers'), fmask_keywords=('weights',))

    def test_value_targets(self):
        from src import Compose, RandomScale, RandomCrop, RandomFlip, RandomBrightnessContrast, \
            RandomGaussianBlur, NormalizeMeanStd

        augmentation_pipeline = Compose([
            RandomScale(scaling_limit=(1.1, 1.6, 0.4, 0.6, 0.4, 0.6), always_apply=True),
            # match the size of nuclei in target data: enlarge in z axis and shrink in x and y axes
            RandomCrop(shape=(32, 320, 320), always_apply=True),  # crop to the desired model input shape
            RandomFlip(axes_to_choose=None, p=1),  # random flipping
            RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, always_apply=True),
            # match intensity characteristics of the target data
            RandomGaussianBlur(max_sigma=(1, 1.5, 1.5), p=0.8),  # match the noise/blur characteristics
            NormalizeMeanStd(mean=35.27, std=27.42, always_apply=True)  # normalize voxel values
        ], value_keywords=('val1', 'val2'))

        value_num = 2
        value_str = 'Some nice image of only ones.'
        input_dict = {'image': np.ones((100, 400, 400)), 'val1': value_num, 'val2': value_str}

        output_dict = augmentation_pipeline(**input_dict)

        self.assertTupleEqual(output_dict['image'].shape, (1, 32, 320, 320))
        self.assertEqual(output_dict['val1'], value_num)
        self.assertEqual(output_dict['val2'], value_str)


if __name__ == '__main__':
    unittest.main()
