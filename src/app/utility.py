import os

from fileManager import FileManager

class Utility:

    file_manager = FileManager()

    def is_image_valid(self, sourceString):
        return (os.path.exists(os.path.join(self.file_manager.dir_in, sourceString)) and (sourceString!=".DS_Store") and (sourceString.split('.')[-1]=='jpg'))

    def is_rectangle_valid(self, box):
        return box[0]>0