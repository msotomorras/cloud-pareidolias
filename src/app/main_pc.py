from evaluate_images import EvaluateImages
from slack_bot import SlackBot

class Main():

    def __init__(self):
        self.evaluate_images = EvaluateImages()
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

    def print_status(self, classImg):
        print('tweet image')
        statusStr = 'Check out this image! I think I can see ' + self.lookup_class(classImg)
        print('status::::', statusStr)

    def main(self):
        classImg, img_id = self.evaluate_images.main()
        if classImg != -1:
            self.slack_bot.start_bot_and_post_img(self.lookup_class(classImg), img_id)
            self.print_status(classImg)

main = Main()
main.main()
