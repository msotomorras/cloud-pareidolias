import numpy as np
import cv2
import os

from utility import Utility
from file_setup import FileSetup
from file_operations import FileOperations
from image_operations import ImageOperations

class ImageProcessing:

    utility = Utility()
    file_setup = FileSetup()
    file_operations = FileOperations()
    image_operations = ImageOperations()

    def generate_region_of_interest (self, img, imgSrc):
        are_there_images_to_evaluate = False
        img = self.image_operations.resize_image(img)
        self.file_operations.save_image(img, imgSrc.replace('.', '_original.'), self.file_setup.dir_debug) 
        mask = self.image_operations.create_mask(img)
        contours = self.image_operations.get_contours(mask)

        # print some outputs for debug
        imgMask = self.image_operations.mask_image(img, mask)
        edges = self.image_operations.get_edges(mask, 10, 200)
        self.file_operations.save_image(mask, imgSrc.replace('.', '_mask.'), self.file_setup.dir_debug)  
        self.file_operations.save_image(imgMask, imgSrc.replace('.', '_masked_img.'), self.file_setup.dir_debug)    
        print('img processed')
        for cnt in contours:
            rectangle = self.image_operations.get_bounding_box(cnt, img)
            if rectangle is not None and self.utility.is_rectangle_valid(rectangle):
                rectangle = self.utility.correct_coords_if_negative(rectangle)
                croppedImg = self.image_operations.crop_image(img, rectangle)
                img_with_bounding_box = self.image_operations.draw_rectangle_on_img (img, imgSrc, rectangle)
                
                self.file_operations.save_image(croppedImg,  imgSrc, self.file_setup.dir_classify)
                self.file_operations.save_image(img_with_bounding_box, imgSrc, self.file_setup.dir_results)
                
                are_there_images_to_evaluate = True
        return are_there_images_to_evaluate 

    def generate_outlined_images (self, classImg):
        for root, dirs, files in os.walk(self.file_setup.dir_classify):  
            for filename in files:
                if self.utility.is_image_valid(filename):
                    img = cv2.imread(os.path.join(self.file_setup.dir_classify, filename), 1)
                    if img is not None:
                        thresLimits = [20,60]
                        if classImg != 1:
                            img = cv2.medianBlur(img,5)
                            img = cv2.medianBlur(img,5)
                        mask = self.image_operations.create_mask(img)
                        img = self.image_operations.mask_image(img, mask)
                        outlinedImg = self.image_operations.get_edges(img, thresLimits[0],thresLimits[1])
                        outlinedImg = cv2.bitwise_not(cv2.cvtColor(outlinedImg,cv2.COLOR_GRAY2BGR))
                        doubleOutput = np.concatenate((outlinedImg, outlinedImg), axis=1)
                        print('Outline generated:', filename)
                        self.file_operations.save_image(doubleOutput, filename, self.file_setup.dir_pix2pix)
                        self.file_operations.save_image(outlinedImg, filename, self.file_setup.dir_classify_outlines)

    def generate_final_image(self):
        for root, dirs, files in os.walk(self.file_setup.dir_results):  
            for filename in files:
                print ('file:', filename)
                original_image = cv2.imread(os.path.join(self.file_setup.dir_results, filename))
                generated_image = cv2.imread(os.path.join(self.file_setup.dir_results_pix2pix, filename.replace('.jpg', '.png')))
                if original_image is not None and generated_image is not None:
                    resultImg = self.image_operations.fit_generated_image_to_original_image(original_image, generated_image)
                    mergedImg = np.concatenate((original_image, resultImg), axis=1)
                    return mergedImg