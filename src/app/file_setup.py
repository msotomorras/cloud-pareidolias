import os
from file_manager import FileManager

class FileSetup:

    file_manager = FileManager()

    def __init__(self):
        self.dir_in = '../../01-InputImages'
        self.dir_classify = '../../02-Classify'
        self.dir_pix2pix = '../../03-Pix2Pix'
        self.dir_out = '../../04-Results'
        self.dir_debug = '../../05-Debug'
        self.dir_classify_outlines = os.path.join(self.dir_classify, 'outlines')
        self.dir_results = os.path.join(self.dir_out, 'results')
        self.dir_results_pix2pix = os.path.join(self.dir_out, 'images')
        self.dir_final = os.path.join(self.dir_out, 'final')

        self.model_classification = '../../classification/model.h5'

        self.model_cats = '../../models/cats2'
        self.model_flowers = '../../models/flowers'
        self.model_pokemons = '../../models/pokemons'

    def setup_file_structure(self):
        self.file_manager.create_dir_if_not_existing(self.dir_classify)
        self.file_manager.create_dir_if_not_existing(self.dir_classify_outlines)
        self.file_manager.create_dir_if_not_existing(self.dir_pix2pix)
        self.file_manager.create_dir_if_not_existing(self.dir_debug)
        self.file_manager.create_dir_if_not_existing(self.dir_out)
        self.file_manager.create_dir_if_not_existing(self.dir_results)
        self.file_manager.create_dir_if_not_existing(self.dir_results_pix2pix)
        self.file_manager.create_dir_if_not_existing(self.dir_final)