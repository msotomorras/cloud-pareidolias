from evaluate_images import EvaluateImages

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

def print_status(classImg):
    print('tweet image')
    statusStr = 'Check out this image! I think I can see ' + lookup_class(classImg)
    print('status::::', statusStr)

def main():
    classImg = evaluate_images.main()
    if classImg:
        print_status(classImg)

main()
