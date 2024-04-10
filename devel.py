# coding: utf-8

from volumentations_biomedicine.core.composition import *
from volumentations_biomedicine.augmentations import *
import numpy as np
import tifffile
import nibabel as nib


if  __name__ == '__main__':
    #path files
    debug = False
    data_samples = ["ekarev2023-crop-small-anisotropic.tif", "ekarev2023-crop-small-isotropic.tif", "brain.nii","Fluo-C3DH-H157_01_t000.tif","ekarev2023-crop2-3-zscaled.tif",
                     "ekarev2023-crop2-2.tif"]
    number_of_sample = 0

    if debug:
        path_to_image = "./Data_samples/" + data_samples[number_of_sample] 
        path_to_augumented_images = "./data_augumented/"
    else:
        path_to_image = "../Data_samples/" + data_samples[number_of_sample] 
        path_to_augumented_images = "../data_augumented/"
    path_to_augumented_images = "D:/CBIA/augumented/"

# These are functions for loading and saving images - 
# They are not the perfect, feel free to change them according to your needs.

def image_to_numpy(path):
    file_format = path.split(".")[-1]
    if file_format == "tif":
        numpy_image = tifffile.imread(path)
        affine_array = None
    if file_format == "nii":
        img = nib.load(path)
        numpy_image = img.get_fdata()
        affine_array = img.affine
    return numpy_image.astype(np.float32) , affine_array

def numpy_to_file(path, image, affine_array):
    file_format = path.split(".")[-1]
    if file_format == "tif":
        if len(image.shape) == 5:
            tifffile.imsave(path, image,  metadata={'axes': 'TZCYX'}, imagej= True)
        if len(image.shape) == 4:
            tifffile.imsave(path, image,  metadata={'axes': 'ZCYX'}, imagej= True)
        else:
            tifffile.imsave(path, image, imagej= True)
    if file_format == "nii":
        array_img = nib.Nifti1Image(image, affine_array)
        nib.save(array_img, path)


def numpy_remove_negative(image):
    negative_indices = image < 0
    image[negative_indices] = 0
    return image


def numpy_normalization(image):
    ''' 
    0 = min
    1 = max 
   '''
    minimun = image.min()
    maximum = image.max()
    if(minimun < 0):
        image = image + minimun
    image = image / maximum
    return image, maximum, minimun
 

def numpy_reverse_normalization(image, minimum, maximum,normalize_with_regards_to_max):
    '''
    can be negative
    '''
    print("image.min()" +  str(image.min()))
    if image.min() < 0:
        image = image - image.min()
    print("maximum:" + str(maximum) + "  image.max(): " + str(image.max()))
    if image.max() == 0:
        return image.astype(np.ushort)
    elif normalize_with_regards_to_max:
        image = image / (  image.max() / maximum)
    else:
        image =  image * maximum
    return image.astype(np.ushort) 

def convention_format(image, multiple_channels = False):

    shape = list(range(len(image.shape)))  
    shape.reverse() 

    if multiple_channels:
        
        shape.insert(0, shape.pop(2))
        return np.transpose(image, shape)
    else:
        return np.expand_dims(np.transpose(image, shape), axis=0)


def move_channels_back(image):
    
    if image.shape[0] > 1:

        shape = list(range(len(image.shape)))
        shape.reverse()
        shape.insert(len(image.shape) - 3, shape.pop())
        return np.transpose(image, shape)
    else:
        shape = [i - 1 for i in range(len(image.shape))]
        #shape = list(range(len(image.shape)))
        shape.reverse()
        shape.pop()
        return np.transpose(image.squeeze(0), shape)


def image_preparation(path,multipleChannels = False):
    maximum, minimum, affine_array = None, None, None 
    image, affine_array = image_to_numpy(path)
    image = convention_format(image,multipleChannels)
    image, maximum, minimum = numpy_normalization(image)
    return image, maximum, minimum, affine_array

def image_save(path, image, minimum, maximum, affine_array,normalize_with_regards_to_max):
    ##affine_array not sure if needed for .nii file. 
    image = numpy_reverse_normalization(image,minimum, maximum,normalize_with_regards_to_max)
    image = move_channels_back(image)
    numpy_to_file(path,image,affine_array)



def get_augmentation():
    return Compose([
        #RandomFlip(axes_to_choose= [1,2,3], p=1),
        #RandomCrop((200,250,30), p = 1)
        #RandomScale2( scale_limit= (0.9, 1.1), p = 1),
        #AffineTransform( angle_limit= [(25,25), (25,25), (25,25)], border_mode="constant", scaling_coef = [1,1,3.22832576] , scale_back= False  ,p=1), # True, , # 
        #CenterCrop( shape=(200,200,15), ignore_index= 200 ,p=1),
        #Resize(shape= (300,100, 20), ignore_index= 203, p = 1)
        #Scale( scale_factor = 0.5, ignore_index= 203 )
        Pad(pad_size= (8,9), p = 1)
        #NormalizeMeanStd( mean= [0.5, 2 ], std = 0.5  , p= 1),
        #Normalize(mean = 2, std = 0, p = 1)
    ], p=1.0)


def get_augmentationMore():
    return Compose([
        RandomGamma( gamma_limit = (0.8 , 1,2) , p = 0.8),
        RandomRotate90(axes=[3,3,3],p=1),
        GaussianBlur(sigma = 1.2, p = 0.8),
    ],targets= [ ['image' , "image1"] , ['mask'], ['float_mask'] ] , p=1.0)


if  __name__ == '__main__':
    
    name_of_file = path_to_image.split("/")[-1] 
    multipleChannels = True
    normalize_with_regards_to_max = False
    path_to_image1 = "../Data_samples/" + data_samples[3] 
    print(path_to_image)


    img, maximum,minimum, affine_array = image_preparation(path_to_image ,multipleChannels)
    mask = img[0].copy()
    aug = get_augmentation()    
    data = {'image': img, 'mask' : mask  } #   #, 'image1': img1
    aug_data = aug(**data)
    img = aug_data['image']
    mask = aug_data['mask']

    #image_save(path_to_augumented_images + "pad" + name_of_file, img, minimum,maximum,affine_array,normalize_with_regards_to_max)