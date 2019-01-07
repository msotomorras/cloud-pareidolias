import sys
import os
import numpy as np
import cv2
import evaluatepix2pix as pix2pix
from keras.models import load_model
import tensorflow as tf

from utility import Utility
from fileManager import FileManager
from fileOperations import FileOperations
from imageProcessing import ImageProcessing

class EvaluateImages:

    utility = Utility()
    file_manager = FileManager()
    file_operations = FileOperations()
    image_processing = ImageProcessing()

    file_manager.setup_file_structure()

    def get_new_region_of_interest_from_image(self):
        imgList = os.listdir(self.file_manager.dir_in)
        nImgs = len(imgList)
        for i in range(nImgs):
            new_roi_from_image = False
            if self.utility.is_image_valid(imgList[i]):
                img = cv2.imread(os.path.join(FileManager.dir_in, imgList[i]), 1)
                new_roi_from_image = self.image_processing.generate_region_of_interest(img, str(imgList[i]))
                return new_roi_from_image
            else:
                return new_roi_from_image

    def classify_images(self):
        graph1 = tf.Graph()
        with graph1.as_default():
            session1 = tf.Session()
            with session1.as_default():
                classifier = load_model(self.file_manager.model_classification)
                for root, dirs, files in os.walk(self.file_manager.dir_classify):  
                    for filename in files:
                        if self.utility.is_image_valid(filename):
                            img = cv2.imread(os.path.join(self.file_manager.dir_classify, 'outlines', filename), 0)
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
            pix2pix.evaluatePix2pix(self.file_manager.model_cats, 1.25/1)
        elif classImg == 1:
            pix2pix.evaluatePix2pix(self.file_manager.model_flowers, 1.25/1)
        else:
            pix2pix.evaluatePix2pix(self.file_manager.model_pokemons, 1.25/1)

    def main(self):
        there_is_new_roi = self.get_new_region_of_interest_from_image()
        if there_is_new_roi:
            self.image_processing.generate_outlined_images(1)
            classImg = self.classify_images()
            print('detected class: ', classImg)
            self.correct_outlines(classImg)
            self.evaluate_pix2pix(classImg)
            self.file_operations.save_final_image()
            return classImg

