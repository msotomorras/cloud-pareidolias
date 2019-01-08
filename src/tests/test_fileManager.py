import sys
import os
import numpy as np
sys.path.append("../app")

from file_setup import FileSetup

class TestFileStructure:

    def test_creates_dir_classify_if_not_existing(self):
        file_setup = FileSetup()
        file_setup.setup_file_structure()
        assert os.path.exists(file_setup.dir_classify)

    def test_creates_dir_classify_outlines_if_not_existing(self):
        file_setup = FileSetup()
        file_setup.setup_file_structure()
        assert os.path.exists(file_setup.dir_classify_outlines)

    def test_creates_dir_pix2pix_if_not_existing(self):
        file_setup = FileSetup()
        file_setup.setup_file_structure()
        assert os.path.exists(file_setup.dir_pix2pix)

    def test_creates_dir_debug_if_not_existing(self):
        file_setup = FileSetup()
        file_setup.setup_file_structure()
        assert os.path.exists(file_setup.dir_debug)

    def test_creates_dir_out_if_not_existing(self):
        file_setup = FileSetup()
        file_setup.setup_file_structure()
        assert os.path.exists(file_setup.dir_out)

    def test_creates_dir_results_if_not_existing(self):
        file_setup = FileSetup()
        file_setup.setup_file_structure()
        assert os.path.exists(file_setup.dir_results)

    def test_creates_dir_final_if_not_existing(self):
        file_setup = FileSetup()
        file_setup.setup_file_structure()
        assert os.path.exists(file_setup.dir_final)

    def saves_file_in_classify_dir(self):
        img = np.zeros([100,100,3], dtype=np.uint8)

        file_setup = FileSetup()
        file_setup.save_image(img, 'white.jpg', 'classify')
        assert os.path.isfile(os.path.join(file_setup.dir_classify, 'white.jpg'))

