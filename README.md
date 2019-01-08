# cloud-pareidolias

Pareidolia is a psychological phenomenon in which the mind responds to a stimulus, usually an image or a sound, by perceiving a familiar pattern where none exists. Like seeing a face in this picture:

  ![pareidolia](https://www.artnews.com/wp-content/uploads/2017/08/4689253598_ccaa7fe938_b.jpg)

# Project

This project uses `OpenCV` , `tensorflow` and `keras` to find pareidolias in clouds, create an image of the imagined shape, and letting us know via twitter.

**For now**, the code is trained to recognize and imagine **three** different classes: **cats**, **flowers** and **pokemons**. Here is how these clouds would look like as flowers and as cats:

  ![](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/04-Results/example%20pics/merged.jpg)

## Overview

The pipeline for the project looks as follows:
* Takes a picture of the sky as **input**
* **Process** the picture
   * Search for clouds in the image
      * Thresholds image to find white areas
      * Looks for areas within a defined threshold
      * Substracts the background in the original image to generate outlines
      * Generates the outlines of the image
   * Classifies the outlines of the cloud using `keras`
   * Generates drawing according to the detected class using `tensorflow` and `pix2pix`
   * Generates a final image composed by the original picture of the sky and the imagined shape
* **Share** the processed and generated picture in twitter

## Dependencies

I'm running the code in a virtual environment in mac and in the raspberry PI, that contains the following:

* **Python 3.5.6**, modern version of **numpy**, **argparse** and other modules that can be installed via `pip` and will be included in `requirements.text`
* **OpenCV 3.4.4**. In order to install it in the mac, I followed [this website](https://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/).<br/>
For installing it in the PI, I followed [this](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi) repo.
* **tensorflow 1.12.0** , It can be installed via `pip`. You can follow [these instructions](https://www.tensorflow.org/install/pip)
* **keras 2.2.4** with tensorflow backend, can be installed via `pip` for mac.
* **Twython** to tweet new images when found

## Getting started

### Running the code

The models for running the pix2pix are not included in the repo due to their size. I used [this colab](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/pix2pix/pix2pix_eager.ipynb) script in order to get my model trained. Some helpful datasets can be found [here](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/)
1. **Get the code**. `git clone` the repo and install dependecies
2. **Get models for pix2pix**
3. **Define your input images** by pasting them into the 01-InputImages folder
4. **Run your code and check your results** by running `python main.py` in `/src/app/` directory
5. **Define twitter auth keys** to tweet a new state and image if found

### Running the tests

1. Install [pytest](https://docs.pytest.org/en/latest/getting-started.html)
2. Go to `/src/tests` directory and run `pytest` command

## Pipeline details

### 1. Take a picture of the sky

   ![mask img](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/05-Debug/img_0.jpg)
       
### 2. Process the picture

**1. Search for clouds in the image - Get Region of Interest (ROI)**

   * Thresholds image to find white areas - Generate mask
   
      ![mask img](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/05-Debug/img_0_mask.jpg)
   
   * Looks for areas within a defined threshold  - Find interesting areas in the picture and extract them
   
      ![bounding box](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/02-Classify/img_0.jpg)
   
**2. Classify the image - what shape does it resemble to?**

   * Generate outlines
    
      ![outlines](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/02-Classify/outlines/img_0.jpg)
   
   * Classify image
   
      In this case it sees a **pokemon**!

**3. Generates drawing according to the detected class**

   ![pix2pix](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/04-Results/images/img_0.png)
   
**4. Merge results and generate new image**
   
   ![final](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/04-Results/final/final_img_0.jpg)


### 3. Tweet the image!

   ![tweet](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/04-Results/final/tweet.jpg)
