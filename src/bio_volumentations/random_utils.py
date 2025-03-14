# ============================================================================================= #
#  Author:       Pavel Iakubovskii, ZFTurbo, ashawkey, Dominik Müller,                          #
#                Samuel Šuľan, Lucia Hradecká, Filip Lux                                        #
#  Copyright:    albumentations:    : https://github.com/albumentations-team                    #
#                Pavel Iakubovskii  : https://github.com/qubvel                                 #
#                ZFTurbo            : https://github.com/ZFTurbo                                #
#                ashawkey           : https://github.com/ashawkey                               #
#                Dominik Müller     : https://github.com/muellerdo                              #
#                Lucia Hradecká     : lucia.d.hradecka@gmail.com                                #
#                Filip Lux          : lux.filip@gmail.com                                       #
#                                                                                               #
#  Volumentations History:                                                                      #
#       - Original:                 https://github.com/albumentations-team/albumentations       #
#       - 3D Conversion:            https://github.com/ashawkey/volumentations                  #
#       - Continued Development:    https://github.com/ZFTurbo/volumentations                   #
#       - Enhancements:             https://github.com/qubvel/volumentations                    #
#       - Further Enhancements:     https://github.com/muellerdo/volumentations                 #
#       - Biomedical Enhancements:  https://gitlab.fi.muni.cz/cbia/bio-volumentations           #
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

from .biovol_typing import TypeSextetFloat, TypeTripletFloat


np_rng = np.random.default_rng()  # random number generator used in all the functions below


def uniform(low=0.0, high=1.0, size=None):
    """Draw samples from a uniform distribution.
    Samples are uniformly distributed over the half-open interval [low, high).

    If size is None, returns a single value (or an array if low/high are arrays).
    """
    return np_rng.uniform(low, high, size)


def sample_range_uniform(limits: TypeSextetFloat) -> TypeTripletFloat:
    """Draw samples from a uniform distribution.
    Samples are uniformly distributed over the half-open interval [low, high).
    """
    return (np_rng.uniform(limits[0], limits[1]),
            np_rng.uniform(limits[2], limits[3]),
            np_rng.uniform(limits[4], limits[5]))


def random(size=None):
    """Return random floats in the half-open interval [0.0, 1.0).

    If size is None, return a single value.
    """
    return np_rng.random(size)


def normal(mean=0.0, std=1.0, size=None):
    """Draw random samples from a normal (Gaussian) distribution.

    If size is None, returns a single value (or an array if mean/std are arrays).
    """
    return np_rng.normal(loc=mean, scale=std, size=size)


def poisson(lam=1.0, size=None):
    """Draw samples from a Poisson distribution.

    If size is None, returns a single value (or an array if lam is an array).
    """
    return np_rng.poisson(lam=lam, size=size)


def randint(start, stop, size=None):
    """Draw random integer from the closed interval [start, stop].

    If size is None, returns a single value (or an array if start/stop are arrays).
    """
    return np_rng.integers(low=start, high=stop, endpoint=True, size=size)


def shuffle(seq, **params):
    """Shuffle the sequence in place.
    The order of sub-arrays is changed but their contents remains the same.
    """
    return np_rng.shuffle(seq, **params)


def sample(population, k, replacement=False, shuffle=False, **params):
    """Return a k-length list of elements chosen from the population sequence.
    """
    return np_rng.choice(a=population, size=k, replace=replacement, shuffle=shuffle, **params)

