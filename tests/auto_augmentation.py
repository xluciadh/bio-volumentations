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

import random
import numpy as np

from src.bio_volumentations import AffineTransform, RandomBrightnessContrast, HistogramEqualization, Compose

# You should show that Bio-Volumentations works with existing automatic augmentation frameworks
# such as "AutoAugment: Learning Augmentation Strategies from Data"


def get_func(func_name, param_value=None, p=1, fillcolor=128):
    """
    Return an instantiated transformation object.

    We only use the transforms implemented in Bio-Volumentations: translation, rotation, brightness, contrast, and
    histogram equalization.
    Compared to the original AutoAugment/RandAugment papers, we are missing shearing, color modification transforms 
    (color, posterize, solarize, invert), auto-contrast, and sharpening. These transformations have no counterparts
    in Bio-Volumentations and their use generally makes no sense for biomedical images.
    """

    tr_list = {
        'translateX': AffineTransform(translation=(0, 0, param_value), ival=fillcolor, p=p),
        'translateY': AffineTransform(translation=(0, param_value, 0), ival=fillcolor, p=p),
        'rotate': AffineTransform(angles=(param_value, 0, 0), ival=fillcolor, p=p),
        'contrast': RandomBrightnessContrast(brightness_limit=0, contrast_limit=(param_value, param_value), p=p),
        'brightness': RandomBrightnessContrast(brightness_limit=(param_value, param_value), contrast_limit=0, p=p),
        'equalize': HistogramEqualization(p=p),
    }
    return tr_list[func_name]


#####################################################################################################
#                                                                                                   #
#                                       AUTOAUGMENT                                                 #
#                                                                                                   #
# Paper: https://arxiv.org/abs/1805.09501                                                           #
#                                                                                                   #
# Adapted from unofficial implementation @ https://github.com/DeepVoltaire/AutoAugment/tree/master  #
#                                                                                                   #
#####################################################################################################


def get_func_param_AA(func_name, param_idx):
    """
    Return transformation parameter value.
    """

    ranges = {
        'translateX': np.linspace(-100, 100, 10),
        'translateY': np.linspace(-100, 100, 10),
        'rotate': np.linspace(-30, 30, 10),
        'contrast': np.linspace(-0.9, 0.9, 10),
        'brightness': np.linspace(-30, 30, 10),
        'equalize': [0] * 10,
    }

    return ranges[func_name][param_idx]


def get_policy_AA(p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=128):
    """
    Instantiate two transforms and return them in a list ( = an AutoAugment policy).
    """

    param_value = get_func_param_AA(operation1, magnitude_idx1)
    tr1 = get_func(operation1, param_value=param_value, p=p1, fillcolor=fillcolor)

    param_value = get_func_param_AA(operation2, magnitude_idx2)
    tr2 = get_func(operation2, param_value=param_value, p=p2, fillcolor=fillcolor)

    return [tr1, tr2]


class AACIFAR10Policy(object):
    """ Randomly choose a policy.

        The policy list contains 4 of the best 25 Sub-policies on CIFAR10
        and a couple more policies inspired by the top-25 list.
    """
    def __init__(self, fillcolor=128, **params):
        self.policies = [
            Compose(get_policy_AA(0.7, 'rotate', 2, 0.3, 'translateX', 9, fillcolor), **params),
            Compose(get_policy_AA(0.6, 'equalize', 5, 0.5, 'equalize', 1, fillcolor), **params),
            Compose(get_policy_AA(0.5, 'translateX', 8, 0.2, 'equalize', 0, fillcolor), **params),
            Compose(get_policy_AA(0.4, 'translateY', 3, 0.2, 'equalize', 0, fillcolor), **params),
            Compose(get_policy_AA(0.2, 'equalize', 0, 0.6, 'contrast', 1, fillcolor), **params),
            Compose(get_policy_AA(0.2, 'equalize', 0, 0.6, 'contrast', 8, fillcolor), **params),
            Compose(get_policy_AA(0.2, 'equalize', 8, 0.6, 'equalize', 4, fillcolor), **params),
            Compose(get_policy_AA(0.9, 'translateY', 9, 0.7, 'translateY', 9, fillcolor), **params),
            Compose(get_policy_AA(0.7, 'translateY', 9, 0.9, 'contrast', 2, fillcolor), **params),
            Compose(get_policy_AA(0.7, 'translateY', 9, 0.9, 'contrast', 7, fillcolor), **params),
            Compose(get_policy_AA(0.7, 'translateY', 9, 0.9, 'brightness', 7, fillcolor), **params)
        ]

    def __call__(self, **data):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](**data)


def run_AA():
    # Fetch a data sample
    #   img : np.array of shape (3, 182, 600, 600)
    #   mask : np.array of shape (182, 600, 600)
    #   keypoints: list of 3D coordinates
    sample = {'image': np.random.random((3, 182, 600, 600)),
              'mask': np.random.randint(0, 5, (182, 600, 600)),
              'keypoints': [(0, 0, 0), (100, 300, 300), (10, 500, 12)]}

    # Get the policy
    policy = AACIFAR10Policy()

    # Transform the sample
    transformed_sample = policy(**sample)

    assert tuple(transformed_sample['image'].shape) == (3, 182, 600, 600)  # image shape must not change
    assert len(transformed_sample['keypoints']) <= 3  # we can lose some keypoints due to translation/rotation


#####################################################################################################
#                                                                                                   #
#                                       RANDAUGMENT                                                 #
#                                                                                                   #
# Paper: https://arxiv.org/abs/1909.13719                                                           #
#                                                                                                   #
# Adapted from unofficial implementation @ https://github.com/ildoonet/pytorch-randaugment          #
#                                                                                                   #
#####################################################################################################


def augment_list_RA():
    return [
        ('equalize', 0, 1),
        ('rotate', -30, 30),
        ('contrast', -0.9, 0.9),
        ('brightness', -30, 30),
        ('translateX', -100., 100),
        ('translateY', -100., 100),
    ]


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list_RA()

    def __call__(self, **data):
        # Get a list of transforms to use
        ops = random.choices(self.augment_list, k=self.n)
        trs = []
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            trs.append(get_func(op, val))

        # Apply them and return the result
        transform_pipeline = Compose(trs)
        return transform_pipeline(**data)


def run_RA():
    # Fetch a data sample
    #   img : np.array of shape (3, 182, 600, 600)
    #   mask : np.array of shape (182, 600, 600)
    #   keypoints: list of 3D coordinates
    sample = {'image': np.random.random((3, 182, 600, 600)),
              'mask': np.random.randint(0, 5, (182, 600, 600)),
              'keypoints': [(0, 0, 0), (100, 300, 300), (10, 500, 12)]}

    # Get the augmenter
    augmenter = RandAugment(n=3, m=20)

    # Transform the sample
    transformed_sample = augmenter(**sample)

    assert tuple(transformed_sample['image'].shape) == (3, 182, 600, 600)  # image shape must not change
    assert len(transformed_sample['keypoints']) <= 3  # we can lose some keypoints due to translation/rotation


if __name__ == '__main__':

    exception_counter = 0
    total = 10
    for _ in range(total):
        try:
            # run_AA()
            run_RA()
        except Exception as e:
            exception_counter += 1
            print(f'Exception encountered: {e}')

    print(f'{total-exception_counter}/{total} runs successful')
