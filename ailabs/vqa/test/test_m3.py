from unittest import TestCase
import unittest

from ..src.metrics import M3, MeanMeter, MovingMeanMeter


class TestM3(TestCase):

    def setUp(self) -> None:
        super().setUp()

    def test_add_meter(self):
        m3 = M3()
        key = 'train_loss'
        meter = MeanMeter()
        self.assertEqual(m3.add_meter(key, meter), meter)

    def test_to_dict(self):
        m3 = M3()
        key = 'val_loss'
        meter = m3.add_meter(key, MeanMeter())
        print(m3.to_dict())

    def test_encode(self):
        pass

    def test_decode(self):
        pass


if __name__ == '__main__':
    unittest.main()
