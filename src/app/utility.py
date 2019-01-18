import os
import datetime

from file_setup import FileSetup

class Utility:

    def __init__(self):
        self.file_setup = FileSetup()

    def is_image_valid(self, sourceString):
        return (os.path.exists(os.path.join(self.file_setup.dir_in, sourceString)) and (sourceString!='.DS_Store') and (sourceString.split('.')[-1]=='jpg'))

    def correct_coords_if_negative(self, box):
        new_box = []
        for i in range (len(box)):
            new_box.append(max(0,box[i]))
        return new_box

    def is_rectangle_valid(self, box):
        return len(box)>1

    def ignore_ds_store(self, filename):
        return (filename != '.DS_Store')

    def get_date_id(self):
        now = datetime.datetime.now()
        hour = now.hour
        minute = now.minute
        day = now.day
        month = now.month

        composed_date_id = str(day)+str(month)+str(hour)+str(minute)

        return composed_date_id