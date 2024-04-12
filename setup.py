# ============================================================================================= #
#  Author:       Samuel Šuľan, Lucia Hradecká, Filip Lux                                        #
#  Copyright:    Lucia Hradecká     : lucia.d.hradecka@gmail.com                                #
#                Filip Lux          : lux.filip@gmail.com                                       #
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

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

setup(
    name='bio_volumentations',
    version='1.1.0',
    author='Samuel Šuľan, Lucia Hradecká, Filip Lux',
    packages=find_packages(),
    url='https://gitlab.fi.muni.cz/cbia/bio-volumentations',
    description='Library for 3D augmentations of multi-dimensional biomedical images',
    keywords='image,augmentation,volumetric,bioimage,biomedical,preprocessing,transformation',
    long_description_content_type='text/markdown',
    long_description='''Bio-Volumentations is an image augmentation and preprocessing package for 3D, 4D, and 5D 
        (volumetric, multi-channel, and time-lapse) biomedical images.

        This library offers a wide range of image transformations implemented efficiently for large-volume image data. 
        This includes both preprocessing transformations (such as intensity normalisation, padding, and type casting) and 
        augmentation transformations (such as affine transform, noise addition and removal, and contrast manipulation).

        The Bio-Volumentations library is a suitable tool for data manipulation in machine learning applications. 
        It can be used with any major Python deep learning library, including PyTorch, PyTorch Lightning, TensorFlow, and Keras.

        This library builds upon wide-spread libraries such as Albumentations (see the Contributions section below) 
        in terms of design and user interface. Therefore, it can easily be adopted by developers.
    
        More details can be found at the project's GitLab (https://gitlab.fi.muni.cz/cbia/bio-volumentations)
        or at the documentation pages.
        
        The Bio-Volumentations library was inspired by:
        
        - Albumentations:           https://github.com/albumentations-team/albumentations
        - 3D Conversion:            https://github.com/ashawkey/volumentations
        - Continued Development:    https://github.com/ZFTurbo/volumentations
        - Enhancements:             https://github.com/qubvel/volumentations
        - Further Enhancements:     https://github.com/muellerdo/volumentations
        
        The Bio-Volumentations library is distributed under the MIT License.
        
        Copyright (c) 2024 Samuel Šuľan, Lucia Hradecká, Filip Lux
        ''',
    install_requires=[
        'numpy',
        'scipy',
        'scikit-image',
        'matplotlib',
        'SimpleITK',
    ],
    entry_points={
        'console_scripts': [
            'bioVolumentation=bio_volumentations:__init__'
        ]
    }
)
