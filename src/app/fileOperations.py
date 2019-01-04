import cv2
import os
import numpy as np

from fileManager import FileManager
from imageOperations import ImageOperations

class FileOperations:

    file_manager = FileManager()
    image_operations = ImageOperations()

    def save_image (self, img, name, folder):
        cv2.imwrite(os.path.join(folder, name), img)

    def save_final_image (self):
        print('saving final images')
        for root, dirs, files in os.walk(self.file_manager.dir_results):  
            for filename in files:
                print ('file:', filename)
                original_image = cv2.imread(os.path.join(self.file_manager.dir_results, filename))
                generated_image = cv2.imread(os.path.join(self.file_manager.dir_results_pix2pix, filename.replace('.jpg', '.png')))
                if original_image is not None and generated_image is not None:
                    resultImg = self.image_operations.fit_generated_image_to_original_image(original_image, generated_image)
                    mergedImg = np.concatenate((original_image, resultImg), axis=1)
                    self.save_image(mergedImg, 'final_' + filename, self.file_manager.dir_final)