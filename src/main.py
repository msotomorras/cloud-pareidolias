# from picamera import PiCamera
import subprocess
import evaluateInputImages as evaluateImages
# from twython import Twython
import time

APP_KEY = 'Vf6ivBsbe3mUh8q0vNHKBaoqx'
APP_SECRET = 'vlgq8LFeZG2wSG2TGgTcseZIHeaAWFvZli8ZDlo0TvI4F2P9y4'
OAUTH_TOKEN = '1054771303192952837-fKyJbghHpEbXN57HPieLnp83289U2u'
OAUTH_TOKEN_SECRET = 'NcHU52ObP7kD2rb1S9h0sps4yM2SmtMI8szU0vk7O3rl0'

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
    # cmd = "raspistill -vf -o /home/pi/tensorflow/clouds/01-InputImages/img_20.jpg"
    # subprocess.call(cmd, shell=True)
    print('take pic')

def tweet_image(classImg):
    print('tweet image')
    # twitter = Twython(APP_KEY, APP_SECRET,
    #               OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    # photo = open('/home/pi/tensorflow/clouds/04-Results/final/img_20.jpg', 'rb')
    # response = twitter.upload_media(media=photo)
    statusStr = 'Checkout this image! I think I see ' + lookup_class(classImg)
    print('status::::', statusStr)
    # twitter.update_status(status=statusStr, media_ids=[response['media_id']])

def main():
    take_picture()
    time.sleep(5)
    classImg = evaluateImages.main()
    tweet_image(classImg)

main()
