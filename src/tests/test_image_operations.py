import sys
import os
import numpy as np 
import cv2
sys.path.append("../app")

from image_operations import ImageOperations

class TestImageOperations:

    image_operations = ImageOperations()
    img = np.zeros([5, 5, 1], dtype = np.uint8)
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
    
    def test_get_total_area_img(self):
        expected_area = self.img.shape[0]*self.img.shape[1]
        area = self.image_operations.get_total_area_img(self.img)
        assert area == expected_area

    def test_fit_generated_image_to_original_image(self):
        img = self.img
        generated_image = np.zeros([1, 1, 1], dtype = np.uint8)
        generated_image.fill(255)
        margin = 20
        new_generated_image = np.zeros([img.shape[0], margin*2 + generated_image.shape[1]], dtype = np.uint8)
        new_generated_image = cv2.cvtColor(new_generated_image, cv2.COLOR_GRAY2BGR)

        result = self.image_operations.fit_generated_image_to_original_image(img, generated_image)
        assert result.shape == new_generated_image.shape
    

