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

utility = Utility()
file_manager = FileManager()
file_operations = FileOperations()
image_processing = ImageProcessing()

file_manager.setup_file_structure()

class EvaluateImages:

    def get_new_region_of_interest_from_image(self):
        imgList = os.listdir(file_manager.dir_in)
        nImgs = len(imgList)
        for i in range(nImgs):
            new_roi_from_image = False
            if utility.is_image_valid(imgList[i]):
                img = cv2.imread(os.path.join(FileManager.dir_in, imgList[i]), 1)
                new_roi_from_image = image_processing.generate_region_of_interest(img, str(imgList[i]))
                return new_roi_from_image
            else:
                return new_roi_from_image

    def classify_images(self):
        graph1 = tf.Graph()
        with graph1.as_default():
            session1 = tf.Session()
            with session1.as_default():
                classifier = load_model(file_manager.model_classification)
                imgList = os.listdir(os.path.join(file_manager.dir_classify, 'outlines'))
                nImgs = len(imgList)
                print('found %s images' %nImgs)
                for i in range (nImgs):
                    if i>0 and utility.is_image_valid(imgList[i]):
                        img = cv2.imread(os.path.join(file_manager.dir_classify, 'outlines', imgList[i]), 0)
                        img = cv2.resize(img, (64,64))
                        data = img.reshape(1,64,64,1)
                        model_out = classifier.predict(data)
                        return np.argmax(model_out)

    def correct_outlines(self, classImg):
        if classImg == 0:
            image_processing.generate_outlined_images(0) 
        elif classImg == 1:
            image_processing.generate_outlined_images(1) 
        else:
            image_processing.generate_outlined_images(2) 

    def evaluate_pix2pix(self, classImg):
        if classImg == 0:
            pix2pix.evaluatePix2pix(file_manager.model_cats, 1.25/1)
        elif classImg == 1:
            pix2pix.evaluatePix2pix(file_manager.model_flowers, 1.25/1)
        else:
            pix2pix.evaluatePix2pix(file_manager.model_pokemons, 1.25/1)

    def main(self):
        classImg = -1
        there_is_new_roi = self.get_new_region_of_interest_from_image()
        print('new roi', there_is_new_roi)
        if there_is_new_roi:
            image_processing.generate_outlined_images(1)
            classImg = self.classify_images()
            print('detected class: ', classImg)
            self.correct_outlines(classImg)
            self.evaluate_pix2pix(classImg)
            file_operations.save_final_image()
        return 1

x = EvaluateImages()
print('FINISHED PROCESS!: ', x.main())
