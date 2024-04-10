# ============================================================================================= #
#  Author:       Samuel Šulan, Lucia Hradecká, Filip Lux                                        #
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
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='volumentations',
    version='1.0.5',
    author='Roman Sol (ZFTurbo), ashawkey, qubvel, muellerdo, Lucia Hradecka, Samuel Šulan, Filip Lux',
    packages=['volumentations', 'volumentations/augmentations', 'volumentations/conversion', 'volumentations/core'],
    url='https://gitlab.fi.muni.cz/cbia/bio-volumentations',
    description='Library for 3D augmentations for biomedical images',
    long_description='Library for 3D augmentations for biomedical images. Inspired by albumentations.'
                     'More details: https://gitlab.fi.muni.cz/cbia/bio-volumentations',
    install_requires=[
        'scikit-image',
        'scipy',
        'opencv-python',
        'numpy',
        'SimpleITK'
    ],
)
