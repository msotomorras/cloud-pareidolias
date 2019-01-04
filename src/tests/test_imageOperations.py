import sys
import os
import numpy as np 
sys.path.append("../app")

from imageOperations import ImageOperations

class TestImageOperations:

    image_operations = ImageOperations()
    img = np.zeros([5,5,1], dtype = np.uint8)
    box = [1, 1, 3, 3]
    img[box[1]:box[3], box[0]:box[2]] = 255

    def test_crop_image(self):
        img_expected = np.zeros([2,2,1], dtype = np.uint8)
        img_expected.fill(255)
        img_cropped = self.image_operations.crop_image(self.img, self.box)
        assert  np.array_equal(img_expected, img_cropped)

    def test_resize_image(self):
        img_expected = np.zeros([450,600], dtype=np.uint8)
        img_resized = self.image_operations.resize_image(self.img)
        assert np.array_equal(img_resized.shape, img_expected.shape)
