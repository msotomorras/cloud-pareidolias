# cloud-pareidolias

Pareidolia is a psychological phenomenon in which the mind responds to a stimulus, usually an image or a sound, by perceiving a familiar pattern where none exists. Like seeing a face in this picture:

![pareidolia](https://www.artnews.com/wp-content/uploads/2017/08/4689253598_ccaa7fe938_b.jpg)

# Project

This project uses computer vision, keras and tensorflow to find pareidolias in clouds, create an image of the imagined shape, and letting us know via twitter.

## Overview

The pipeline for the project looks as follows:
* Takew a picture of the sky as **input**
* **Process** the picture
   * Looks for a cloud in the image
   * Classifies the outlines of the cloud
   * Generates image of the class detected with the outlines of the cloud
   * Generates a final image composed by the original picture of the sky and the imagined shape
* **Share** the processed and generated picture in twitter

## Dependencies

I'm running the code in a virtual environment in mac, that contains the following:

* **Python 3.5.6**, modern version of **numpy**, **argparse** and other modules that can be installed via `pip` and will be included in `requirements.text`
* **OpenCV 3.4.4**. In order to install it in the mac, I followed [this website](https://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/)
* **tensorflow 1.12.0** , It can be installed via `pip`. You can follow [these instructions](https://www.tensorflow.org/install/pip)
* **keras 2.2.4** with tensorflow backend, can also be installed via `pip`


## 1. Take a picture of the sky

   ![mask img](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/05-Debug/img_0.jpg)
       
## 2. Process the picture

**1. Get Region of Interest (ROI)**

   * Generate mask
   
      ![mask img](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/05-Debug/img_0_mask.jpg)
   
   * Find interesting areas in the picture and extract them
   
      ![bounding box](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/02-Classify/img_0.jpg)
   
**2. Classify the image - what shape does it resemble to?**

   * Generate outlines
    
      ![outlines](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/02-Classify/outlines/img_0.jpg)
   
   * Classify image
   
      In this case it sees a pokemon!

**3. Generate drawing upon found class**

   ![pix2pix](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/04-Results/images/img_0.png)
   
**4. Merge results and generate new image**
   
   ![final](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/04-Results/final/final_img_0.jpg)


## 3. Tweet the image!
