import os
import numpy as np

class FileManager:

    def create_dir_if_not_existing(self, input_dir):
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
            

    

