# Volumentations 3D

Image augmentation package for 3D/4D + channels volume data inspired by albumentations.
**Main assumption** is that the channel is first dimension and afterwards there are spatial dimensions and optionally time if image is 4D.


If input/image is 3D image without channels, then Compose automatically increase input by one dimension which represents one channel. It is recommended to use transformations inside Compose.   


# Usage and testing

**Old**: Download the repository and try inside demo.py or use the approach from ./wrapper/bio_volumentations_wrapper.py

**New**: install as package from pip using
```python
pip install bio-volumentations
```
and import to your project as
```python
import bio_volumentations as biovol
```

## Requirements

NumPy       https://numpy.org/ <br> 
SciPy       https://scipy.org/ <br>
scikit-mage https://scikit-image.org/ <br>
SimpleITK   https://simpleitk.org/ <br>


### Simple Example

```python

def get_augmentation():
    return Compose([
        RandomGamma( gamma_limit = (0.8, 1,2) , p = 0.8),
        RandomRotate90(axes = [1,2,3], p = 1),
        GaussianBlur(sigma = 1.2, p = 0.8)
    ], p=1.0)

aug = get_augmentation()

img = np.random.rand(1, 128, 256, 256)
lbl = np.random.randint(0, 1, size=(128, 256, 256), dtype=np.uint8)

# with mask
data = {'image': img, 'mask': lbl}
aug_data = aug(**data)
img, lbl = aug_data['image'], aug_data['mask']

# without mask
data = {'image': img}
aug_data = aug(**data)
img = aug_data['image']

```


### Simple Example with multiple images

```python

def get_augmentation():
    return Compose([
        RandomGamma( gamma_limit = (0.8, 1,2) , p = 0.8),
        RandomRotate90(axes = [1,2,3], p = 1),
        GaussianBlur(sigma = 1.2, p = 0.8)
    ], p=1.0, 
    targets= [ ['image' , "image1"] , ['mask'], ['float_mask'] ])

aug = get_augmentation()

img = np.random.rand(1, 128, 256, 256)
img1 = np.random.rand(1, 128, 256, 256)
lbl = np.random.randint(0, 1, size=(128, 256, 256), dtype=np.uint8)

# 2 images and mask
data = {'image': img, 'image1': img1, 'mask': lbl}
aug_data = aug(**data)
img, img1, lbl = aug_data['image'],aug_data['image1'], aug_data['mask']


```

### Implemented 3D augmentations

```python
AffineTransform 
CenterCrop 
Contiguous 
Flip 
GaussianBlur 
GaussianNoise 
HistogramEqualization 
Normalize 
NormalizeMeanStd 
Pad 
RandomBrightnessContrast 
RandomCrop 
RandomFlip 
RandomGamma 
RandomGaussianBlur 
RandomRotate90 
RandomScale 
Resize 
Scale
```


## Speed table

Speed in seconds averaged over 100 samples. Test with 1.5X means that output image is 1.5X bigger.

| Aug name | Cube = 64px | Cube = 128px | Cube = 256px | Shape = (512px,512px,64px) |
|----------|-------------|-------------|--------------|--------------|
| Rotate 22.5 degrees | 0.040  | 0.328 | 3.035 | 2.633 |
| AffineTransform | 0.042 | 0.339  | 3.613  | 2.756  |
| Flip | 0.001 | 0.006 | 0.059 | 0.060 |
| GaussianBlur | 0.001 | 0.006 | 0.053 | 0.054 | 
| GaussianNoise | 0.008 | 0.061 | 0.497 | 0.498 |  
| HistogramEqualization | 0.016 | 0.125 | 1.021 | 1.021 |  
| Normalize| 0.002 | 0.023 | 0.199 | 0.198 |  
| NormalizeMeanStd  | 0.001 | 0.006 | 0.052 | 0.052 |  
| RandomBrightnessContrast | 0.001 | 0.004 | 0.033| 0.035 |  
| RandomFlip| 0.001 | 0.007  |0.067 | 0.067 | 
| RandomGamma | 0.003 | 0.027 | 0.229 | 0.229  |  
| RandomGaussianBlur |0.006 | 0.221 | 0.733 | 0.692 |    
| RandomRotate90 | 0.002 | 0.081  | 0.236 | 0.189 |  
| RandomScale | 0.017 | 0.141 | 1.046 | 1.024 |  
| Scale 1.5X | 0.034 | 0.271 | 2.200 | 2.196 |  
| Scale 0.75X | 0.005 | 0.039 | 0.322 | 0.325 | 
| CenterCrop 1.5X | 0.003 | 0.031 | 0.220 | 0.255 |  
| CenterCrop 0.75X | 0.001 | 0.005 | 0.042 |  0.038 |  
| RandomCrop 1.5X | 0.004 | 0.031 | 0.223 | 0.243 |  
| RandomCrop 0.75X | 0.001 | 0.005 | 0.040 | 0.038 |  
| Resize 1.5X  | 0.065 | 0.511 | 4.369 | 4.259 |  
| Resize 0.75X | 0.009 | 0.070 | 0.572 | 0.568 |  


(Rotate is little bit slower than rotate from previous fork)

## Citation

TODO
