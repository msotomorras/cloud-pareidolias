

import os
import time
import re
import random
from file_setup import FileSetup
from slackclient import SlackClient
from slack_keys import SlackKeys

class SlackBot:

    def __init__(self):
        self.channel = 'CFA3CNPMH'
        self.slack_keys = SlackKeys()
        self.slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))
        self.starterbot_id = None
        self.file_setup = FileSetup()

    def get_random_sentence(self, classImg):
        sentence_id = random.randint(1, 3)
        if sentence_id == 1:
            return 'Check out this image! I think I can see ' + classImg + ' in the sky'
        elif sentence_id == 2:
            return 'Look, there’s a cloud with the shape of ' + classImg + ', check it out!'
        elif sentence_id == 3:
            return 'I found ' + classImg + ' in the sky!'

    def post_image(self, classImg, img_id):
        print('post image')
        status = self.get_random_sentence(classImg)
        with open(os.path.join(self.file_setup.dir_final, img_id+'.png'), 'rb') as f:
            contents = f.read()
            self.slack_client.api_call(
                "files.upload",
                channels=self.channel,
                initial_comment=status,
                file=contents,
                title=classImg + ' in the clouds'
            )

    def start_bot_and_post_img(self, classImg, img_id):
        if self.slack_client.rtm_connect(with_team_state=False):
            print("Starter Bot connected and running!")
            starterbot_id = self.slack_client.api_call("auth.test")["user_id"]
            self.post_image(classImg, img_id)
        else:
            print("Connection failed. Exception traceback printed above.")