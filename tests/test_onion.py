import unittest
from image_lens import onion


class Lines(unittest.TestCase):
    def test_bands_read_as_still_moving_or_unknown(self):
        result = [dict(band='맨 위', y=[0, 10], dx=20., dy=0., features=30), dict(band='위', y=[10, 20], dx=3., dy=-2., features=30),
                  dict(band='가운데', y=[20, 30], dx=None, dy=None, features=2)]
        line, picture = onion.lines(result, 'x-onion.jpg')
        self.assertIn('맨 위 오른쪽 20px', line); self.assertIn('위 그대로', line); self.assertIn('가운데 모름', line)
        self.assertIn('x-onion.jpg', picture)

    def test_box_is_padded_and_clamped(self):
        self.assertEqual(onion._box((0, 100, 100, 200), 150, 205), (0, 90, 110, 205))

if __name__ == '__main__': unittest.main()
