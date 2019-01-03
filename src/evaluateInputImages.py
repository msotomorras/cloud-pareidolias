import sys
import os
import numpy as np
import cv2
import argparse
import classifier as classifier
import random
import evaluatepix2pix as pix2pix
from keras.models import load_model
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser(description='input images -> evaluated contours')
    parser.add_argument('--dir_in', default= '../01-InputImages', dest='dir_in', help='directory for input images')
    parser.add_argument('--dir_classify', default= '../02-Classify',dest='dir_classify', help='directory for classify images')
    parser.add_argument('--dir_pix2pix', default= '../03-Pix2Pix',dest='dir_pix2pix', help='directory for pix2pix images')
    parser.add_argument('--dir_out', default= '../04-Results',dest='dir_out', help='directory for results')
    parser.add_argument('--dir_debug', default= '../05-Debug', dest='dir_debug', help='directory for all output images')
    args = parser.parse_args()
    return args

args = parse_args()

imgList = os.listdir(args.dir_in)
nImgs = len(imgList)
print('images %s' % nImgs)
dir_results = os.path.join(args.dir_out, 'results')


def is_image_valid(sourceString):
    if (os.path.exists(os.path.join(args.dir_in, sourceString)) and (sourceString!=".DS_Store") and (sourceString.split('.')[-1]=='jpg')):
        return True
    else:
        return False

# IMAGE OPERATIONS

def crop_image(img, box):
    croppedImg = img[box[1]:box[3], box[0]:box[2]]
    return croppedImg  

def resize_image(imgItem, i):
    img = cv2.imread(os.path.join(args.dir_in, imgItem), 1)
    img = cv2.resize(img,(600,450))
    return img

def threshold_mage(img):
    img = cv2.medianBlur(img,5)
    grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(grayscaled,170,255,cv2.THRESH_BINARY)
    return th

def create_mask (img):
    mask = threshold_mage(img)
    return mask

def mask_image (img, mask):
    imask = mask>0
    whiteMask = np.zeros_like(img, np.uint8)
    whiteMask[imask] = img[imask]
    h, s, v = cv2.split(whiteMask)
    emptyImg = np.full((img.shape[0], img.shape[1], 3), 255, np.uint8)
    imgMasked = cv2.bitwise_and(img,img, dst=emptyImg, mask=v)
    return imgMasked

def get_contours(img):
    im2, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_edges (img, th1, th2):
    return cv2.Canny( img, th1, th2 )


# FILE OPERATIONS

def save_image (img, name, type):
    if (type == 'classify'):
        cv2.imwrite(os.path.join(args.dir_classify, name), img)
    if (type == 'classifyOutlines'):
        cv2.imwrite(os.path.join(args.dir_classify,'outlines', name), img)
    elif (type =='pix2pix'):
        cv2.imwrite(os.path.join(args.dir_pix2pix, name), img)
    elif (type == 'debug'):
        cv2.imwrite(os.path.join(args.dir_debug, name), img)
    elif (type == 'final'):
        cv2.imwrite(os.path.join(args.dir_out,'final', name), img)
    else:
        cv2.imwrite(os.path.join(dir_results, name), img)

def generate_final_image ():
    for root, dirs, files in os.walk(dir_results):  
        for filename in files:
            if filename[0] == '0':
                print ('file:', filename, filename[0])
                imgOriginal = cv2.imread(os.path.join(dir_results, filename))
                filename = filename.split('_', 1)[-1]
                drawing = cv2.imread(os.path.join(args.dir_out, 'images', filename.replace('.jpg', '.png')))
                print('result image dir:', os.path.join(args.dir_out, 'images', filename.replace('.jpg', '.png')))
                if imgOriginal is not None and drawing is not None:
                    marginTop = int((imgOriginal.shape[0]-drawing.shape[0])*0.5)
                    marginLeft = 20
                    resultImg = np.full((imgOriginal.shape[0], drawing.shape[1]+marginLeft*2, 3), 255, np.uint8)
                    resultImg[marginTop:marginTop+drawing.shape[0], marginLeft:marginLeft+drawing.shape[1]] = drawing
                    mergedImg = np.concatenate((imgOriginal, resultImg), axis=1)
                    save_image(mergedImg, 'final_' + filename, 'final')


# SEGMENTATION OF CLOUDS

def get_bounding_box(cnt, img, imgSrc):
    margin = 20
    box = [0]
    if 5000<cv2.contourArea(cnt)<205000:
        (x,y,w,h) = cv2.boundingRect(cnt)
        aspect_ratio = 0.75
        h = int(w*aspect_ratio)
        box = [x-margin, y-margin, x+w+margin, y+h+margin]
    return box

def rectangle_valid(box):
    return box[0]>0

def draw_rectangle_on_img(img, imgSrc, box):
    img2 = img.copy()
    img2 = cv2.rectangle(img2, (box[0], box[1]), (box[2], box[3]), (255,0,0), 2)
    save_image(img2, '0_'+imgSrc, 'results')

# INTERMEDIATE IMAGE GENERATION PROCESSES

def generate_cropped_image (img, mask, imgMasked, imgSrc):
    are_there_images_to_evaluate = 0
    edges = get_edges(mask, 10, 200)
    contours = get_contours(mask)
    print('img processed')
    for cnt in contours:
        rectangle = get_bounding_box(cnt, img, imgSrc)
        if rectangle_valid(rectangle):
            croppedImg = crop_image(img, rectangle)
            save_image(croppedImg,  imgSrc, 'classify')
            draw_rectangle_on_img (img, imgSrc, rectangle)
            are_there_images_to_evaluate = 1
    return are_there_images_to_evaluate 

def generate_outlined_images (classImg):
    for root, dirs, files in os.walk(args.dir_classify):  
        for filename in files:
            if is_image_valid(filename):
                img = cv2.imread(os.path.join(args.dir_classify, filename), 1)
                if img is not None:
                    thresLimits = [20,60]
                    if classImg != 1:
                        img = cv2.medianBlur(img,5)
                        img = cv2.medianBlur(img,5)
                    mask = create_mask(img)
                    img = mask_image(img, mask)
                    outlinedImg = get_edges(img, thresLimits[0],thresLimits[1])
                    outlinedImg = cv2.bitwise_not(cv2.cvtColor(outlinedImg,cv2.COLOR_GRAY2BGR))
                    doubleOutput = np.concatenate((outlinedImg, outlinedImg), axis=1)
                    print('Outline generated:', filename)
                    save_image(doubleOutput, filename, 'pix2pix')
                    save_image(outlinedImg, filename, 'classifyOutlines')

# MAIN FUNCTIONS

def are_there_new_images():
    for i in range(nImgs):
            if is_image_valid(imgList[i]):
                img = resize_image(imgList[i], i)
                mask = create_mask(img)
                save_image(mask, str(i)+'_mask_'+str(imgList[i]), 'debug')
                imgMask = mask_image(img, mask)
                save_image(imgMask, str(i)+'_'+str(imgList[i]), 'debug')
                is_there_new_image = generate_cropped_image(img, mask, imgMask, str(imgList[i]))
                return is_there_new_image

def classify_images():
    graph1 = tf.Graph()
    with graph1.as_default():
        session1 = tf.Session()
        with session1.as_default():
            classifier = load_model('../classification/model.h5')
            imgList = os.listdir(os.path.join(args.dir_classify, 'outlines'))
            nImgs = len(imgList)
            print('found %s images' %nImgs)
            for i in range (nImgs):
                if i>0 and is_image_valid(imgList[i]):
                    img = cv2.imread(os.path.join(args.dir_classify, 'outlines', imgList[i]), 0)
                    img = cv2.resize(img, (64,64))
                    data = img.reshape(1,64,64,1)
                    model_out = classifier.predict(data)
                    return np.argmax(model_out)

def correct_outlines(classImg):
    if classImg == 0:
        generate_outlined_images(0) 
    elif classImg == 1:
        generate_outlined_images(1) 
    else:
        generate_outlined_images(2) 

def evaluate_pix2pix(classImg):
    if classImg == 0:
        pix2pix.evaluatePix2pix('../models/cats2', 1.25/1)
    elif classImg == 1:
        pix2pix.evaluatePix2pix('../models/flowers', 1.25/1)
    else:
        pix2pix.evaluatePix2pix('../models/pokemons', 1.25/1)

def main():
    classImg = -1
    if are_there_new_images():
        generate_outlined_images(1)
        classImg = classify_images()
        print('detected class: ', classImg)
        correct_outlines(classImg)
        evaluate_pix2pix(classImg)
        generate_final_image()
    return classImg

    



