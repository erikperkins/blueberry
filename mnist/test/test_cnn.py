import unittest
from mnist.cnn import digit_image

class TestClassifier(unittest.TestCase):
  def test_digit_image(self):
    image = digit_image(1)
    self.assertIsNotNone(image)

if __name__ == '__main__':
  unittest.main()
