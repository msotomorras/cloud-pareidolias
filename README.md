# cloud-pareidolias

Pareidolia is a psychological phenomenon in which the mind responds to a stimulus, usually an image or a sound, by perceiving a familiar pattern where none exists. Like this face:

![pareidolia](https://www.artnews.com/wp-content/uploads/2017/08/4689253598_ccaa7fe938_b.jpg)

This project focuses on finding pareidolias in clouds, and the process it follows to do that is the following:
(I'll add pics of each step so it's easier to follow)

1. Take a picture

2. Evaluate if there is an interesting shape of a cloud in a picture and segment it:
    * Threshold image 
    * Get mask of the image
    * Get masked image
    * Get bounding box of the region of interest
    
3. Classify the cloud and find out what the algorithm thinks it looks like
4. Generate a drawing of how the algorithm imagines this thing would look in the cloud. 
5. Tweet image so you can get a notification when there's an interesting cloud to look at in the sky!
    
