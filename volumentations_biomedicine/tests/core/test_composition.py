import unittest
import numpy as np
from volumentations_biomedicine.core.composition import Compose
import volumentations_biomedicine.augmentations.transforms as transforms


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


if __name__ == '__main__':
    unittest.main()
