import os
import numpy as np

class FileManager:

    dir_in = '../../01-InputImages'
    dir_classify = '../../02-Classify'
    dir_pix2pix = '../../03-Pix2Pix'
    dir_out = '../../04-Results'
    dir_debug = '../../05-Debug'
    dir_classify_outlines = os.path.join(dir_classify, 'outlines')
    dir_results = os.path.join(dir_out, 'results')
    dir_results_pix2pix = os.path.join(dir_out, 'images')
    dir_final = os.path.join(dir_out, 'final')

    model_classification = '../../classification/model.h5'

    model_cats = '../../models/cats2'
    model_flowers = '../../models/flowers'
    model_pokemons = '../../models/pokemons'

    def setup_file_structure(self):
        self.create_dir_if_not_existing(self.dir_classify)
        self.create_dir_if_not_existing(self.dir_classify_outlines)
        self.create_dir_if_not_existing(self.dir_pix2pix)
        self.create_dir_if_not_existing(self.dir_debug)
        self.create_dir_if_not_existing(self.dir_out)
        self.create_dir_if_not_existing(self.dir_results)
        self.create_dir_if_not_existing(self.dir_results_pix2pix)
        self.create_dir_if_not_existing(self.dir_final)

    def create_dir_if_not_existing(self, input_dir):
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
        
            

    

