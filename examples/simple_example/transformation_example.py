# ============================================================================================= #
#  Author:       Filip Lux, Lucia Hradecká, Jakub Polonský                                      #
#  Copyright:    Filip Lux          : lux.filip@gmail.com                                       #
#                Lucia Hradecká     : lucia.d.hradecka@gmail.com                                #
#                Jakub Polonský                                                                 #
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


##############################################
#      Bio-Volumentations                    #
##############################################

def get_transformation_pipeline():
    return bv.Compose([
        bv.RandomAffineTransform(scaling_limit=1.2, angle_limit=(10, 20, 20), spacing=(1, 0.09, 0.09), p=1),
        bv.CenterCrop(shape=(20, 400, 400), p=1),
        bv.GaussianNoise(var_limit=(0, 75), p=1)
    ])


def transform_sample(sample: dict, transformation_pipeline: bv.Compose):
    return transformation_pipeline(**sample)


##############################################
#      Data handling routines                #
##############################################


def read_sample(input_dir='.'):
    img = tifffile.imread(os.path.join(input_dir, 'image.tif'))
    mask = tifffile.imread(os.path.join(input_dir, 'segmentation_mask.tif'))
    keypoints = read_keypoints()
    bboxes = read_bboxes()
    return img, mask, keypoints, bboxes


def save_sample(img: np.ndarray, mask: np.ndarray, keypoints_t: list, bboxes_t: list, output_dir='.'):
    tifffile.imwrite(os.path.join(output_dir, 'image_transformed.tif'), img)
    tifffile.imwrite(os.path.join(output_dir, 'segmentation_mask_transformed.tif'), mask)
    save_keypoints(keypoints_t, output_dir)
    save_bboxes(bboxes_t, output_dir)


def read_keypoints(in_dir='.'):
    with open(os.path.join(in_dir, 'keypoints.txt'), 'r') as f:
        lines = f.readlines()

    return [tuple(float(coord.strip()) for coord in line.split(',')) for line in lines]


def save_keypoints(kpts_list: list, output_dir='.'):
    lines = [f'{z},{y},{x}\n' for z, y, x in kpts_list]

    with open(os.path.join(output_dir, 'keypoints_transformed.txt'), 'w') as f:
        f.writelines(lines)


def read_bboxes(in_dir='.'):
    def format_item(item: str):
        if item.startswith('('):
            return tuple(float(coord) for coord in item[1:-1].split(','))
        return float(item.strip())

    with open(os.path.join(in_dir, 'bboxes.txt'), 'r') as f:
        lines = f.readlines()

    return [[format_item(item) for item in line.split(';')] for line in lines]


def save_bboxes(bbxs_list: list, output_dir='.'):
    def format_item(item):
        if isinstance(item, tuple):
            return '(' + ','.join(str(v) for v in item) + ')'
        return str(item)

    lines = [
        ';'.join(format_item(item) for item in bbox) + '\n'
        for bbox in bbxs_list
    ]

    with open(os.path.join(output_dir, 'bboxes_transformed.txt'), 'w') as f:
        f.writelines(lines)


##############################################
#      Running the example script            #
##############################################

if __name__ == '__main__':

    # Read data
    img, mask, keypoints, bboxes = read_sample()

    # Create transformation pipeline
    transformation_pipeline = get_transformation_pipeline()

    # Transform data
    sample = {'image': img, 'mask': mask, 'keypoints': keypoints, 'bboxes': bboxes}
    transformed_sample = transform_sample(sample, transformation_pipeline)
    img_t = transformed_sample['image']
    mask_t = transformed_sample['mask']
    keypoints_t = transformed_sample['keypoints']
    bboxes_t = transformed_sample['bboxes']

    # Save results
    save_sample(img_t, mask_t, keypoints_t, bboxes_t)

    print('Results were stored to the current directory: see files "image_transformed.tif", '
          '"segmentation_mask_transformed.tif", "keypoints_transformed.txt", "bboxes_transformed.txt"')
