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
    author='Lucia Hradecká, Filip Lux, Samuel Šuľan',
    packages=find_packages(),
    url='https://gitlab.fi.muni.cz/cbia/bio-volumentations',
    description='Library for 3D augmentations of multi-dimensional biomedical images',
    long_description='Library for preprocessing and augmentation of 3D biomedical images. The library can handle 3D-5D images: volumetric, multi-channel, and time-lapse.\n'
                     'Inspired by:\n'
                     '- Albumentations:           https://github.com/albumentations-team/albumentations\n'
                     '- 3D Conversion:            https://github.com/ashawkey/volumentations\n'
                     '- Continued Development:    https://github.com/ZFTurbo/volumentations\n'
                     '- Enhancements:             https://github.com/qubvel/volumentations\n'
                     '- Further Enhancements:     https://github.com/muellerdo/volumentations\n\n'
                     'More details here: https://gitlab.fi.muni.cz/cbia/bio-volumentations.',
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
