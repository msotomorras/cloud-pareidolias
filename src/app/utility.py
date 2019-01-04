import os

from fileManager import FileManager

class Utility:

    file_manager = FileManager()

    def is_image_valid(self, sourceString):
        return (os.path.exists(os.path.join(self.file_manager.dir_in, sourceString)) and (sourceString!=".DS_Store") and (sourceString.split('.')[-1]=='jpg'))

    def correct_coords_if_negative(self, box):
        new_box = []
        for i in range (len(box)):
            new_box.append(max(0,box[i]))
        return new_box

    def is_rectangle_valid(self, box):
        return box[0]>0