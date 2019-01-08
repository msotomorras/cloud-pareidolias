import sys
import os
import numpy as np 
import cv2
sys.path.append("../app")

from image_processing import ImageProcessing

class TestImageProcessing:

    image_processing = ImageProcessing()
    img = np.zeros([5, 5, 3], dtype = np.uint8)
    box = [1, 1, 3, 3]
    img[box[1]:box[3], box[0]:box[2]] = 255

    def test_generate_region_of_interest(self):
        expected_is_there_new_roi = True
        is_there_new_roi = self.image_processing.generate_region_of_interest(self.img, 'test.jpg')
        assert  is_there_new_roi == expected_is_there_new_roi
