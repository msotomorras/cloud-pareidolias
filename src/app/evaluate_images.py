import sys
import os
import numpy as np
import cv2
from keras.models import load_model
import tensorflow as tf

from utility import Utility
from file_setup import FileSetup
from file_operations import FileOperations
from image_processing import ImageProcessing
import evaluate_pix2pix as pix2pix

class EvaluateImages:

    utility = Utility()
    file_setup = FileSetup()
    file_operations = FileOperations()
    image_processing = ImageProcessing()

    file_setup.setup_file_structure()

    def get_new_region_of_interest_from_image(self):
        for root, dirs, files in os.walk(self.file_setup.dir_in): 
            for filename in files:
                if self.utility.ignore_ds_store(filename):
                    new_roi_from_image = False 
                    if self.utility.is_image_valid(filename):
                        img = cv2.imread(os.path.join(self.file_setup.dir_in, filename), 1)
                        new_roi_from_image = self.image_processing.generate_region_of_interest(img, filename)
                        return new_roi_from_image
                    else:
                        return new_roi_from_image

    def classify_images(self):
        graph1 = tf.Graph()
        with graph1.as_default():
            session1 = tf.Session()
            with session1.as_default():
                classifier = load_model(self.file_setup.model_classification)
                for root, dirs, files in os.walk(self.file_setup.dir_classify):  
                    for filename in files:
                        if self.utility.is_image_valid(filename):
                            img = cv2.imread(os.path.join(self.file_setup.dir_classify, 'outlines', filename), 0)
                            img = cv2.resize(img, (64,64))
                            data = img.reshape(1,64,64,1)
                            model_out = classifier.predict(data)
                            return np.argmax(model_out)

    def correct_outlines(self, classImg):
        if classImg == 0:
            self.image_processing.generate_outlined_images(0) 
        elif classImg == 1:
            self.image_processing.generate_outlined_images(1) 
        else:
            self.image_processing.generate_outlined_images(2) 

    def evaluate_pix2pix(self, classImg):
        if classImg == 0:
            pix2pix.evaluatePix2pix(self.file_setup.model_cats, 1.25/1)
        elif classImg == 1:
            pix2pix.evaluatePix2pix(self.file_setup.model_flowers, 1.25/1)
        else:
            pix2pix.evaluatePix2pix(self.file_setup.model_pokemons, 1.25/1)

    def save_final_image(self):
        merged_img = self.image_processing.generate_final_image()
        self.file_operations.save_image(merged_img, 'final_' + 'img_final.jpg', self.file_setup.dir_final)

    def main(self):
        there_is_new_roi = self.get_new_region_of_interest_from_image()
        if there_is_new_roi:
            self.image_processing.generate_outlined_images(1)
            classImg = self.classify_images()
            print('detected class: ', classImg)
            self.correct_outlines(classImg)
            self.evaluate_pix2pix(classImg)
            self.save_final_image()
            return classImg