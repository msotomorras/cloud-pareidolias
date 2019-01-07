# cloud-pareidolias

Pareidolia is a psychological phenomenon in which the mind responds to a stimulus, usually an image or a sound, by perceiving a familiar pattern where none exists. Like seeing a face in this picture:

![pareidolia](https://www.artnews.com/wp-content/uploads/2017/08/4689253598_ccaa7fe938_b.jpg)

This project focuses on finding pareidolias in clouds, following this process:
### 1. Take a picture of the sky

   ![ img](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/05-Debug/img_0.jpg | width=100 )
       
### 2. Process the picture

### 3. Make a tweet if there was a shape found

-----
## 1. Take a picture of the sky

## 2. Process picture:
**2.1. Find Region Of Interest (ROI)**
    * Generate mask of the image

   In this step we do image thresholding, in order to find big areas of white pixels in the sky. The brightest areas in the picture would become white pixels, and the rest of the areas would be black. 
    ![mask img](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/05-Debug/img_0_mask.jpg)

   * Find areas within a defined threshold and extract that part of the image<br/>
    Defining the areas that we would be interested in analysing and extract them from the picture to continue the process<br/>
    ![bounding box](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/04-Results/results/img_0.jpg)

   * Generate outlines of the ROI <br/>

   ![bounding box](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/02-Classify/img_0.jpg)
   ![outlines](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/02-Classify/outlines/img_0.jpg)

**2.2. Classify image**

   Here we send the outlines of our cloud to the classifier algorithm. In this example, it thinks it resembles a pokemon!

**2.3. Generate pix2pix<br/>**

   ![pix2pix](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/04-Results/images/img_0.png)

**2.4. Create final image**
    And here is the result!<br/>
    ![final image](https://raw.githubusercontent.com/msotomorras/cloud-pareidolias/master/04-Results/final/final_img_0.jpg)


## 3. Tweet picture

If a cloud with a shape has been found, tweet image so you can get a notification. Go see if you agree!
    

