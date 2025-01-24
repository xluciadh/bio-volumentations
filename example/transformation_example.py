# ============================================================================================= #
#  Author:       Filip Lux, Lucia Hradecká,                                                     #
#  Copyright:    Filip Lux          : lux.filip@gmail.com                                       #
#                Lucia Hradecká     : lucia.d.hradecka@gmail.com                                #
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


import numpy as np
import tifffile
import os
import bio_volumentations as bv


# transformation pipeline using BioVolumentations
def get_transformation_pipeline():
    return bv.Compose([
        bv.RandomAffineTransform(scaling_limit=2, angle_limit=(10, 30, 30), spacing=(1, 0.09, 0.09), p=1),
        bv.CenterCrop(shape=(20, 400, 400), p=1),
        bv.GaussianNoise(var_limit=(0, 75), p=1)
    ])


def transform_sample(sample: dict, transformation_pipeline: bv.Compose):
    return transformation_pipeline(**sample)


# image handling routines
def read_sample(input_dir: str):
    img = tifffile.imread(os.path.join(input_dir, 'image.tif'))
    mask = tifffile.imread(os.path.join(input_dir, 'segmentation_mask.tif'))
    return img, mask
    
    
def save_sample(img: np.ndarray, mask: np.ndarray, output_dir: str):
    tifffile.imwrite(os.path.join(output_dir, 'image_transformed.tif'), img)
    tifffile.imwrite(os.path.join(output_dir, 'segmentation_mask_transformed.tif'), mask)


if __name__ == '__main__':
    img, mask = read_sample('.')

    transformation_pipeline = get_transformation_pipeline()

    sample = {'image': img, 'mask': mask}
    transformed_sample = transform_sample(sample, transformation_pipeline)
    img_t = transformed_sample['image']
    mask_t = transformed_sample['mask']

    save_sample(img_t, mask_t, '.')
    print('Samples "image_transformed.tif" and "segmentation_mask_transformed.tif" were stored to this directory.')
    
