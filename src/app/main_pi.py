# All commented lines must be uncommented for the code to run in the pi

from picamera import PiCamera
import subprocess
from twython import Twython
import time
from slack_bot import SlackBot
from evaluate_images import EvaluateImages
from twitter_keys import TwitterKeys

class MainPi():
    def __init__(self):
        self.evaluate_images = EvaluateImages()
        self.twitter_keys = TwitterKeys()
        self.slack_bot = SlackBot()

    def lookup_class(self, classImg):
        class_name = []
        if classImg == 0:
            class_name = 'cats'
        elif classImg == 1:
            class_name = 'flowers'
        else:
            class_name ='a pokemon'
        return class_name
        
    def take_picture(self):
        cmd = "raspistill -vf -o /home/pi/cloud-pareidolias/01-InputImages/img_20.jpg"
        subprocess.call(cmd, shell=True)
        print('take pic')

    def tweet_image(self, classImg):
        print('tweet image')
        twitter = Twython(self.twitter_keys.APP_KEY, self.twitter_keys.APP_SECRET,
                    self.twitter_keys.OAUTH_TOKEN, self.twitter_keys.OAUTH_TOKEN_SECRET)
        photo = open('/home/pi/cloud-pareidolias/04-Results/final/img_20.jpg', 'rb')
        response = twitter.upload_media(media=photo)
        statusStr = 'Check out this image! I think I can see ' + self.lookup_class(classImg)
        print('status::::', statusStr)
        twitter.update_status(status=statusStr, media_ids=[response['media_id']])

    def print_status(self, classImg):
        print('tweet image')
        statusStr = 'Check out this image! I think I can see ' + self.lookup_class(classImg)
        print('status::::', statusStr)

    def main(self):
        self.take_picture()
        time.sleep(5)
        classImg, img_id = self.evaluate_images.main()
        if classImg:
            self.slack_bot.start_bot_and_post_img(self.lookup_class(classImg), img_id)
            self.print_status(classImg)

main_pi = MainPi()
main_pi.main()
