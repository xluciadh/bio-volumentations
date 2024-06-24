import unittest
import numpy as np
from bio_volumentations.core.composition import Compose
import bio_volumentations.augmentations.transforms as transforms


class TestComposeConversion(unittest.TestCase):
    def test_no_conversion(self):
        sh = (36, 200, 250, 3)
        img = np.zeros(sh)

        tr = Compose([transforms.NormalizeMeanStd(mean=[0],
                                                  std=[0])])
        result = tr(image=img)

        res_image = result['image']
        self.assertIsInstance(res_image, np.ndarray)
        self.assertTrue(np.all(np.equal(img, res_image)))

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
        from bio_volumentations import Compose, RandomGamma, RandomFlip, GaussianBlur

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
        import numpy as np
        from bio_volumentations import Compose, RandomGamma, RandomRotate90, GaussianBlur

        # Create the transformation using Compose from a list of transformations and define targets
        aug = Compose([
            RandomGamma(gamma_limit=(0.8, 1.2), p=0.8),
            RandomRotate90(axes = [1, 2, 3], p = 1),
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


if __name__ == '__main__':
    unittest.main()
