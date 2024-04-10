try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

setup(
    name='bio_volumentations',
    version='1.0.10',
    author='Lucia Hradecka, Filip Lux, Samuel Šuľan',
    packages=find_packages(),
    url='https://gitlab.fi.muni.cz/xsulan/volumentations-biomedicine/',
    description='Library for 3D augmentations for biomedical images',
    long_description='Library for 3D preprocessing and augmentation of biomedical images. Inspired by Albumentations and Volumentations. '
                     'More details: https://gitlab.fi.muni.cz/xsulan/volumentations-biomedicine/',
    install_requires=[
        'scikit-image',
        'scipy',
        'opencv-python',
        "numpy",
    ],
    entry_points={
        'console_scripts': [
            'bioVolumentation=bio_volumentations:__init__'
        ]
    }
)
