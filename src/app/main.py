# All commented lines must be uncommented for the code to run in the pi

# from picamera import PiCamera
import subprocess
# from twython import Twython
import time
import datetime

from evaluateImages import EvaluateImages

APP_KEY = 'Vf6ivBsbe3mUh8q0vNHKBaoqx'
APP_SECRET = 'vlgq8LFeZG2wSG2TGgTcseZIHeaAWFvZli8ZDlo0TvI4F2P9y4'
OAUTH_TOKEN = '1054771303192952837-fKyJbghHpEbXN57HPieLnp83289U2u'
OAUTH_TOKEN_SECRET = 'NcHU52ObP7kD2rb1S9h0sps4yM2SmtMI8szU0vk7O3rl0'

evaluate_images = EvaluateImages()

def lookup_class(classImg):
    class_name = []
    if classImg == 0:
        class_name = 'cats'
    elif classImg == 1:
        class_name = 'flowers'
    else:
        class_name ='a pokemon'
    return class_name

    
def take_picture():
    # cmd = "raspistill -vf -o /home/pi/cloud-pareidolias/01-InputImages/img_20.jpg"
    # subprocess.call(cmd, shell=True)
    print('take pic')

def tweet_image(classImg):
    print('tweet image')
    # twitter = Twython(APP_KEY, APP_SECRET,
    #               OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    # photo = open('/home/pi/cloud-pareidolias/04-Results/final/img_20.jpg', 'rb')
    # response = twitter.upload_media(media=photo)
    statusStr = 'Check out this image! I think I can see ' + lookup_class(classImg)
    print('status::::', statusStr)
    # twitter.update_status(status=statusStr, media_ids=[response['media_id']])

def main():
    print(datetime.datetime.now())
    take_picture()
    # time.sleep(5)
    classImg = evaluate_images.main()
    if classImg:
        tweet_image(classImg)

main()
