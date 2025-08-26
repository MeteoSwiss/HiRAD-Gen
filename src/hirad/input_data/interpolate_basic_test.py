import unittest

class Testing(unittest.TestCase):
    def test_trivial(self):
        a = 1
        b = 1
        self.assertEqual(a, b)

if __name__ == '__main__':
    unittest.main()