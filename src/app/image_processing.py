import numpy as np
import os

from utility import Utility
from file_setup import FileSetup
from cv_operations import CvOperations
from image_operations import ImageOperations

class ImageProcessing:

    def __init__(self):
        self.utility = Utility()
        self.file_setup = FileSetup()
        self.cv_operations = CvOperations()
        self.image_operations = ImageOperations()

    def generate_region_of_interest (self, img, imgSrc):
        are_there_images_to_evaluate = False
        img = self.image_operations.resize_image(img)
        self.cv_operations.save_image(img, imgSrc.replace('.', '_original.'), self.file_setup.dir_debug) 
        mask = self.image_operations.create_mask(img)
        contours = self.image_operations.get_contours(mask)

        # print some outputs for debug
        imgMask = self.image_operations.mask_image(img, mask)
        edges = self.image_operations.get_edges(mask, 10, 200)
        self.cv_operations.save_image(mask, imgSrc.replace('.', '_mask.'), self.file_setup.dir_debug)  
        self.cv_operations.save_image(imgMask, imgSrc.replace('.', '_masked_img.'), self.file_setup.dir_debug)    
        print('img processed')
        for cnt in contours:
            rectangle = self.image_operations.get_bounding_box(cnt, img)
            if rectangle is not None and self.utility.is_rectangle_valid(rectangle):
                rectangle = self.utility.correct_coords_if_negative(rectangle)
                croppedImg = self.image_operations.crop_image(img, rectangle)
                img_with_bounding_box = self.image_operations.draw_rectangle_on_img (img, imgSrc, rectangle)
                
                self.cv_operations.save_image(croppedImg,  imgSrc, self.file_setup.dir_classify)
                self.cv_operations.save_image(img_with_bounding_box, imgSrc, self.file_setup.dir_results)
                
                are_there_images_to_evaluate = True
        return are_there_images_to_evaluate 

    def generate_outlined_images (self, classImg):
        for root, dirs, files in os.walk(self.file_setup.dir_classify):  
            for filename in files:
                if self.utility.is_image_valid(filename):
                    img = self.cv_operations.read_image(filename, self.file_setup.dir_classify)
                    if img is not None:
                        thresLimits = [20,60]
                        img = self.image_operations.blur_image_if_not_flowers(img, classImg)
                        mask = self.image_operations.create_mask(img)
                        img = self.image_operations.mask_image(img, mask)
                        outlinedImg = self.image_operations.get_edges(img, thresLimits[0],thresLimits[1])
                        outlinedImg = self.cv_operations.invert_image(outlinedImg)
                        doubleOutput = np.concatenate((outlinedImg, outlinedImg), axis=1)
                        print('Outline generated:', filename)
                        self.cv_operations.save_image(doubleOutput, filename, self.file_setup.dir_pix2pix)
                        self.cv_operations.save_image(outlinedImg, filename, self.file_setup.dir_classify_outlines)

    def generate_final_image(self):
        for root, dirs, files in os.walk(self.file_setup.dir_results):  
            for filename in files:
                print ('file:', filename)
                original_image = self.cv_operations.read_image(filename, self.file_setup.dir_results)
                generated_image = self.cv_operations.read_image(filename.replace('.jpg', '.png'), self.file_setup.dir_results_pix2pix)
                if original_image is not None and generated_image is not None:
                    resultImg = self.image_operations.fit_generated_image_to_original_image(original_image, generated_image)
                    mergedImg = np.concatenate((original_image, resultImg), axis=1)
                    return mergedImg