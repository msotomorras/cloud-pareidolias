import cv2
import os
from image_operations import ImageOperations

class CvOperations:

    def __init__(self):
        self.image_operations = ImageOperations()

    def read_image(self, filename, folder, type='BGR'):
        if(type == 'BGR'):
            img = cv2.imread(os.path.join(folder, filename), 1)
        else:
            img =  cv2.imread(os.path.join(folder, filename), 0)
        return self.image_operations.resize_image(img)

    def to_gray_scale(self, img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    def to_BGR(self, img):
        return cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    def resize(self, img, w, h):
        return cv2.resize(img, (w,h))

    def save_image (self, img, name, folder):
        cv2.imwrite(os.path.join(folder, name), img)

    def blur_image(self, img, times):
        for i in range (times):
            img = cv2.medianBlur(img, 5)
        return img

    def invert_image(self, img):
        if (len(img.shape)==2):
            img = self.to_BGR(img)
        return cv2.bitwise_not(img)

    def save_images(self, img1, name1, folder1, img2, name2, folder2):
        self.save_image(img1, name1, folder1)
        self.save_image(img2, name2, folder2)