import math
import unittest
from image_lens import motif


def unit(values):
    m = sum(values)/len(values); v = [x-m for x in values]; n = math.sqrt(sum(x*x for x in v))
    return [x/n for x in v]


class Assign(unittest.TestCase):
    def test_same_drawings_repeat_and_one_offs_stay_unlabelled(self):
        a, b, c = unit([1, 2, 3, 4]), unit([4, 1, 3, 2]), unit([2, 4, 1, 3])
        windows = [dict(start=a, end=b, frames=15), dict(start=c, end=c, frames=15), dict(start=a, end=b, frames=15),
                   dict(start=a, end=c, frames=15), dict(start=a, end=b, frames=62)]
        self.assertEqual(motif.assign(windows), ['A', None, 'A', None, None])

    def test_missing_thumbs_never_repeat(self):
        a = unit([1, 2, 3, 4])
        self.assertEqual(motif.assign([dict(start=a, end=a, frames=15), dict(start=None, end=a, frames=15)]), [None, None])


if __name__ == '__main__': unittest.main()
