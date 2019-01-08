import cv2
import os
import numpy as np

from file_setup import FileSetup
from image_operations import ImageOperations

class FileOperations:

    def save_image (self, img, name, folder):
        cv2.imwrite(os.path.join(folder, name), img)
        