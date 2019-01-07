# cloud-pareidolias

Pareidolia is a psychological phenomenon in which the mind responds to a stimulus, usually an image or a sound, by perceiving a familiar pattern where none exists. Like seeing a face in this picture:

![pareidolia](https://www.artnews.com/wp-content/uploads/2017/08/4689253598_ccaa7fe938_b.jpg)

This project focuses on finding pareidolias in clouds, and the process to do that is the following:
## Take a picture of the sky

![sky pic] (https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/01-InputImages/img_8.jpg)


## Process picture:
#### Find Region Of Interest (ROI)
1. Generate mask of the image
...In this step we do image thresholding, in order to find big areas of white pixels in the sky. The brightest areas in the picture would become white pixels, and the rest of the areas would be black. 
![mask img](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/05-Debug/img_0_mask.jpg)
2. Find areas within a defined threshold and extract that part of the image
...Defining the areas that we would be interested in analysing and extract them from the picture to continue the process
...![bounding box](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/04-Results/results/img_0.jpg)
![bounding box](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/02-Classify/img_0.jpg)
3. Generate outlines
...![outlines](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/02-Classify/outlines/img_0.jpg)
#### Classify image
Here we send the outlines of our cloud to the classifier algorithm. In this example, it thinks it resembles a pokemon!
#### Generate pix2pix
...![pix2pix](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/04-Results/images/img_0.png)
#### Create final image
And here is the result!
...![final image](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/04-Results/final/final_img_0.jpg)
## Tweet picture
If a cloud with a shape has been found, tweet image so you can get a notification. Go see if you agree!
    
