import cv2
import os

class CvOperations:

    def read_image(self, filename, folder, type='BGR'):
        if(type == 'BGR'):
            return cv2.imread(os.path.join(folder, filename), 1)
        else:
            return cv2.imread(os.path.join(folder, filename), 0)

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