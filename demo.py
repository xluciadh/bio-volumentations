from volumentations_biomedicine.core.composition import *
from volumentations_biomedicine.augmentations import *

from devel import *


def get_augmentation():
    return Compose([
        CenterCrop( shape=(200,200,15) ,p=1),
        AffineTransform( angle_limit = [(20,20), (20,20), (20,20)],  scaling_coef= [1,1,3.22832576], p=1), 
        Pad(pad_size= [10,3,0],  p = 1)
        ], p=1.0)
 


def get_augmentation_more():
    return Compose([
        CenterCrop( shape=(200,200,20) ,p=1),
        AffineTransform( angle_limit = [(20,20), (20,20), (20,20)],  scaling_coef= [1,1,3.22832576], p=1), 
        Pad(pad_size= [10,3,0], p = 1)    ],
        targets= [ ['image' , "image1"] , ['mask'], ['float_mask'] ], p=1.0)



if  __name__ == '__main__':

    #constants
    data_samples = ["ekarev2023-crop-small-anisotropic.tif", "ekarev2023-crop-small-isotropic.tif", "brain.nii","Fluo-C3DH-H157_01_t000.tif","ekarev2023-crop2-3-zscaled.tif",
                     "ekarev2023-crop2-2.tif"]
    number_of_sample = 1
    path_to_image = "../Data_samples/" + data_samples[number_of_sample] 
    path_to_image1 = "../Data_samples/" + data_samples[2] 
    
    path_to_augumented_images = "D:/CBIA/demo/"
    name_of_file = path_to_image.split("/")[-1]
    multipleChannels = True
    normalize_with_regards_to_max = False
    


    img, maximum,minimum, affine_array = image_preparation(path_to_image ,multipleChannels)
    img1, maximum1,minimum1, affine_array1 = image_preparation(path_to_image1 ,False)
    mask = img[0].copy()
    #or use 
    #img = np.random.rand(1, 128, 256, 256) 
    #mask = np.random.randint(0, 1, size=(128, 256, 256), dtype=np.uint8)

    aug = get_augmentation()
    
    #Categorizing data 
    data = {'image': img , 'mask' : mask }   
    #data = {'image': img , 'mask' : mask , 'image1': img1 }   
    
    #Starting transformations
    aug_data = aug(**data)

    #Taking data after transformations
    #img, mask, img1  = aug_data['image'], aug_data['mask'] , aug_data['image1']
    img, mask = aug_data['image'], aug_data['mask'] 
    
    # just for saving purposes
    mask = mask[np.newaxis, :]

    #Saving images
    image_save(path_to_augumented_images + "Image" + ".tif", img, minimum,maximum,affine_array,normalize_with_regards_to_max)
    image_save(path_to_augumented_images + "mask" + ".tif", mask, minimum,maximum,affine_array,normalize_with_regards_to_max)
    #image_save(path_to_augumented_images + "Image1" + ".nii", img1, minimum1,maximum1,affine_array1,normalize_with_regards_to_max)



