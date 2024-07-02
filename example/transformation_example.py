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
    
