import logging

import unittest
import numpy as np

import convert_accumulation

class Testing(unittest.TestCase):
    def test_convert_accumulation(self):
        one_hour = np.array([
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 2, 2, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 3, 5, 2, 0, 0, 0, 0, 0, 0, 0],
            [3, 4, 5, 2, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
        ]).transpose()
        accumulated = np.ones_like(one_hour) * convert_accumulation.NULL_CONST
        for i in range(5, accumulated.shape[0]):
            accumulated[i,:] = np.sum(one_hour[i-5:i+1,:], 0)
        accumulated = accumulated[5:, :]
        
        interpolated = convert_accumulation.interpolate_1h_from_6h(accumulated)
        
        self.assertTrue(np.array_equal(interpolated, one_hour[5:,:]))

        self.assertEqual(convert_accumulation.check_interpolation(interpolated, accumulated), 0)
        
if __name__ == '__main__':
    unittest.main()