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
        contours = self.image_operations.get_contours(img)
        print('img processed')
        for cnt in contours:
            rectangle = self.image_operations.get_bounding_box(cnt, img)
            if rectangle is not None and self.utility.is_rectangle_valid(rectangle):
                self.generate_cropped_and_result_img(img, rectangle, imgSrc)
                are_there_images_to_evaluate = True
        return are_there_images_to_evaluate

    def generate_outlined_images (self, classImg):
        for root, dirs, files in os.walk(self.file_setup.dir_classify):  
            for filename in files:
                if self.utility.is_image_valid(filename):
                    img = self.cv_operations.read_image(filename, self.file_setup.dir_classify)
                    if img is not None:
                        thresLimits = [20,60]
                        background_substraction_img = self.substract_background_from_image(img, classImg)
                        outlinedImg = self.get_outlines(background_substraction_img, thresLimits)
                        self.generate_img_to_classify(outlinedImg, filename)
                        self.generate_img_for_pix2pix(outlinedImg, filename)
                        print('Outline generated:', filename)

    def create_final_image(self):
        for root, dirs, files in os.walk(self.file_setup.dir_results):  
            for filename in files:
                print ('final file:', filename)
                original_image = self.cv_operations.read_image(filename, self.file_setup.dir_results)
                generated_image = self.cv_operations.read_image(filename.replace('.jpg', '.png'), self.file_setup.dir_results_pix2pix)
                if original_image is not None and generated_image is not None:
                    self.generate_final_image(original_image, generated_image)

    def substract_background_from_image(self, img, classImg):
        img = self.image_operations.blur_image_if_not_flowers(img, classImg)
        mask = self.image_operations.create_mask(img)
        img = self.image_operations.mask_image(img, mask)
        return img

    def get_outlines(self, img, thresLimits):
        outlinedImg = self.image_operations.get_edges(img, thresLimits[0],thresLimits[1])
        outlinedImg = self.cv_operations.invert_image(outlinedImg)
        return outlinedImg

    def generate_final_image(self, original_image, generated_image):
        result_img = self.image_operations.fit_generated_image_to_original_image(original_image, generated_image)
        merged_img = np.concatenate((original_image, result_img), axis=1)
        self.cv_operations.save_image(merged_img, 'final_' + 'img_final.jpg', self.file_setup.dir_final)

    def generate_cropped_img_from_roi(self, img, imgSrc, rectangle):
        cropped_img = self.image_operations.crop_image(img, rectangle)
        self.cv_operations.save_image(cropped_img, imgSrc, self.file_setup.dir_classify)

    def generate_original_img_with_bounding_box(self, img, imgSrc, rectangle):
        img_with_bounding_box = self.image_operations.draw_rectangle_on_img (img, imgSrc, rectangle)
        self.cv_operations.save_image(img_with_bounding_box, imgSrc, self.file_setup.dir_results)

    def generate_cropped_and_result_img(self, img, rectangle, imgSrc):
        rectangle = self.utility.correct_coords_if_negative(rectangle)
        self.generate_cropped_img_from_roi(img, imgSrc, rectangle)
        self.generate_original_img_with_bounding_box(img, imgSrc, rectangle)

    def generate_img_to_classify(self, img, filename):
        self.cv_operations.save_image(img, filename, self.file_setup.dir_classify_outlines)

    def generate_img_for_pix2pix(self, img, filename):
        doubleOutput = np.concatenate((img, img), axis=1)
        self.cv_operations.save_image(doubleOutput, filename, self.file_setup.dir_pix2pix)   

    def generate_debug_images(self, img, mask, imgSrc):
        imgMask = self.image_operations.mask_image(img, mask)
        self.cv_operations.save_image(img, imgSrc.replace('.', '_original.'), self.file_setup.dir_debug) 
        self.cv_operations.save_image(mask, imgSrc.replace('.', '_mask.'), self.file_setup.dir_debug)  
        self.cv_operations.save_image(imgMask, imgSrc.replace('.', '_masked_img.'), self.file_setup.dir_debug)            