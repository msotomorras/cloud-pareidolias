import numpy as np
import random
import cv2
import os

class ImageOperations:

    def crop_image(self, img, box):
        print('box', box)
        return img[box[1]:box[3], box[0]:box[2]]  

    def threshold_image(self, img):
        img = cv2.medianBlur(img,5)
        grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, th = cv2.threshold(grayscaled,170,255,cv2.THRESH_BINARY)
        return th

    def create_mask (self, img):
        mask = self.threshold_image(img)
        return mask

    def mask_image (self, img, mask):
        imask = mask>0
        whiteMask = np.zeros_like(img, np.uint8)
        whiteMask[imask] = img[imask]
        h, s, v = cv2.split(whiteMask)
        emptyImg = np.full((img.shape[0], img.shape[1], 3), 255, np.uint8)
        imgMasked = cv2.bitwise_and(img,img, dst=emptyImg, mask=v)
        return imgMasked

    def get_contours(self, img):
        mask = self.create_mask(img)
        im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def get_edges (self, img, th1, th2):
        return cv2.Canny( img, th1, th2 )

    def draw_rectangle_on_img(self, img, imgSrc, box):
        img2 = img.copy()
        img2 = cv2.rectangle(img2, (box[0], box[1]), (box[2], box[3]), (255,0,0), 2)
        return img2
      
    def get_total_area_img(self, img):
        return img.shape[0]*img.shape[1]

    def get_area_thresholds(self, area):
        low_threshold = random.randint(5, 6)*0.01
        high_threshold = random.randint(1, 2)* 0.1
        return [5000, 15000] #[area*low_threshold, area*high_threshold]

    def get_bounding_box(self, cnt, img):
        margin = 20
        box = [0]
        area = self.get_total_area_img(img)
        area_threshold = self.get_area_thresholds(area)
        if area_threshold[0]<cv2.contourArea(cnt)<area_threshold[1]:
            (x,y,w,h) = cv2.boundingRect(cnt)
            aspect_ratio = 0.75
            h = int(w*aspect_ratio)
            box = [x-margin, y-margin, x+w+margin, y+h+margin]
        return box

    def fit_generated_image_to_original_image(self, original_image, generated_image):
        print('height generated shape ', generated_image.shape)
        marginTop = int((original_image.shape[0]-generated_image.shape[0])*0.5)
        print('margin toop', marginTop)
        marginLeft = 20
        resultImg = np.full((original_image.shape[0], generated_image.shape[1]+marginLeft*2, 3), 255, np.uint8)
        resultImg[marginTop:marginTop+generated_image.shape[0], marginLeft:marginLeft+generated_image.shape[1]] = generated_image
        return resultImg

    def blur_image_if_not_flowers(self, img, classImg):
        if classImg != 1:
            img = cv2.medianBlur(img, 5)
            img = cv2.medianBlur(img, 5)
        return img