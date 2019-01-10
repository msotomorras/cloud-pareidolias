

import os
import time
import re
from slackclient import SlackClient

class SlackBot:

    def __init__(self):
        self.channel = 'CFA3CNPMH'
        self.slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))
        self.starterbot_id = None

    def post_image(self, classImg):
        print('post image')
        status = 'Check out this image! I think I can see ' + classImg + ' in the sky'
        with open('../../04-Results/final/img_final.png', 'rb') as f:
            contents = f.read()
            self.slack_client.api_call(
                "files.upload",
                channels=self.channel,
                initial_comment=status,
                file=contents,
                title=classImg + ' in the clouds'
            )

    def start_bot_and_post_img(self, classImg):
        if self.slack_client.rtm_connect(with_team_state=False):
            print("Starter Bot connected and running!")
            starterbot_id = self.slack_client.api_call("auth.test")["user_id"]
            self.post_image(classImg)
        else:
            print("Connection failed. Exception traceback printed above.")