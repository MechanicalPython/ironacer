import unittest
from yolo import BBoxConvert


class TestBBoxConvert(unittest.TestCase):
    def setUp(self):
        self.box = BBoxConvert('Images/', 'Labels/', 'Yolo_labels')

    def test_1(self):
        r = self.box.convert_single_bbox_file_to_yolo('Labels/squirrel_1.txt', 'Images/squirrel_1.jpg')[0]
        c = [0, 0.318359, 0.503472, 0.441406, 0.573611]
        self.assertAlmostEqual(r[1], c[1], 1)
        self.assertAlmostEqual(r[2], c[2], 1)
        self.assertAlmostEqual(r[3], c[3], 1)
        self.assertAlmostEqual(r[4], c[4], 1)


    def test_100(self):
        r = self.box.convert_single_bbox_file_to_yolo('Labels/squirrel_100.txt', 'Images/squirrel_100.jpg')[0]
        c = [0, 0.512109, 0.450000, 0.311719, 0.358333]
        self.assertAlmostEqual(r[1], c[1], 1)
        self.assertAlmostEqual(r[2], c[2], 1)
        self.assertAlmostEqual(r[3], c[3], 1)
        self.assertAlmostEqual(r[4], c[4], 1)

    def test_1000(self):
        r = self.box.convert_single_bbox_file_to_yolo('Labels/squirrel_1000.txt', 'Images/squirrel_1000.jpg')[0]
        c = [0, 0.639844, 0.582639, 0.009375, 0.009722]
        self.assertAlmostEqual(r[1], c[1], 1)
        self.assertAlmostEqual(r[2], c[2], 1)
        self.assertAlmostEqual(r[3], c[3], 1)
        self.assertAlmostEqual(r[4], c[4], 1)

    def test_1001(self):
        r = self.box.convert_single_bbox_file_to_yolo('Labels/squirrel_1001.txt', 'Images/squirrel_1001.jpg')[0]
        c = [0, 0.580078, 0.420833, 0.552344, 0.816667]
        self.assertAlmostEqual(r[1], c[1], 1)
        self.assertAlmostEqual(r[2], c[2], 1)
        self.assertAlmostEqual(r[3], c[3], 1)
        self.assertAlmostEqual(r[4], c[4], 1)


if __name__ == '__main__':
    unittest.main()