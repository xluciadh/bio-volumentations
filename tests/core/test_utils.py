# ============================================================================================= #
#  Author:       Lucia Hradecká                                                                 #
#  Copyright:    Lucia Hradecká     : lucia.d.hradecka@gmail.com                                #
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


import unittest
from src.bio_volumentations.core.utils import are_mutually_disjoint


class KeywordChecks(unittest.TestCase):
    def test_disjoint_1(self):
        l1 = ('a', 'b', 'c')

        self.assertTrue(are_mutually_disjoint(l1))

    def test_disjoint_2(self):
        l1 = ('a', 'b', 'c')
        l2 = ('x', 'y', 'z')

        self.assertTrue(are_mutually_disjoint(l1, l2))

    def test_nondisjoint_2(self):
        l1 = ('a', 'b', 'c')
        l2 = ('x', 'y', 'b')

        self.assertFalse(are_mutually_disjoint(l1, l2))

    def test_disjoint_5(self):
        l1 = ('a', 'b', 'c')
        l2 = ('x', 'y', 'z')
        l3 = ('k', 'l', 'm')
        l4 = ('mask', 'mask2')
        l5 = ('img',)

        self.assertTrue(are_mutually_disjoint(l1, l2, l3, l4, l5))

    def test_nondisjoint_5(self):
        l1 = ('a', 'mask', 'c')
        l2 = ('x', 'y', 'z')
        l3 = ('k', 'l', 'm')
        l4 = ('mask', 'mask2')
        l5 = ('img',)

        self.assertFalse(are_mutually_disjoint(l1, l2, l3, l4, l5))

    def test_nondisjoint_5_multiple(self):
        l1 = ('a', 'mask', 'c')
        l2 = ('x', 'y', 'z')
        l3 = ('c', 'l', 'm', 'x')
        l4 = ('mask', 'mask2', 'a')
        l5 = ('img',)

        self.assertFalse(are_mutually_disjoint(l1, l2, l3, l4, l5))

    def test_repeated_1(self):
        l1 = ('a', 'b', 'c', 'c')

        self.assertFalse(are_mutually_disjoint(l1))

    def test_repeated_2(self):
        l1 = ('a', 'b', 'c', 'c')
        l2 = ('x', 'y', 'z')

        self.assertFalse(are_mutually_disjoint(l1, l2))


if __name__ == '__main__':
    unittest.main()
