## Vehicle Detection Project
### Final project of Self-driving car engineer nanodegree - term 1

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/example_car_notcar.png
[image2]: ./output_images/hog_visualisations.png
[image3]: ./output_images/sliding_window_boxes.png
[image4]: ./output_images/boxes_heatmaps_box.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

The implementation is found in the `Vehicle Detection Project.ipynb` jupyter notebook along with some additional annotation.

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

In part 2 of the notebook (see notebook headings) I started by reading in all the `vehicle` (n=8792) and `non-vehicle` (n=8968).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

In part 4 of the notebook (after defining some helper functions), I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=6`, `pixels_per_cell=8` and `cells_per_block=2`, using `hog_channel=0`.  It helps get a feel for how the HOG representation will identify the gradient patterns which make up a vehicle in an image:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

In part 5 of the notebook, I tried various combinations of parameters and determined that the `YCrCb` colour space worked a lot better than `RGB`.  I also increased `orientations = 9` and used `hog_channel = 'ALL'` to increase the information used to generate the HOG represenation from the image.  I visualised these changes and then used them to train the model and checked the accuracy with each change, finding that these produced a good test accuracy on the dataset.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In part 5 of the notebook, I trained a linear SVM using sklearn's `LinearSVC` with a train/test split of 0.8/0.2 randomly shuffled hog, spatian and histogram features from the car and notcar data sets.  It was important to maintain a common scale for the features using sklearn's `StandardScaler` to ensure none of these sources would dominate during the training of the model.  The output of the training is shown below, achieving a test accuracy of 99.13%

---

118.93804001808167 Seconds to compute features...

Using: 9 orientations 8 pixels per cell and 2 cells per block

Feature vector length: 8460

6.64 Seconds to train SVC...

Test Accuracy of SVC =  0.9913

---

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In part 6 of the notebook, I implemented a basic sliding window search and tried it out on the test images for the project.  I limited the y coordinates for consideration to between 400-656 to exclude the sky and areas of trees where cars would not be found, and eliminate some false positives.  

I experimented with the window scales to try to find one which would be able to find cars both reasonably near the camera and further away, and found a `96,96` window to be a nice compromise on the test images.

Similarly, trying different overlaps revealed that for this window scale, a `0.5` value meant that vehicles larger than the window size would appropriately be found in conjoined windows (to feed into our heat map later).

The output of this section is shown below.  I noted that I had a false-positive identified in the shadows of the top-right test image, but otherwise the performance was quite good.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In part 8 of the notebook, I ultimately searched using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  This section uses `window = 64` size and a `cell_per_step = 2` value which equates to a 75% overlap between windows to ensure a nice dense detection around vehicles.

Using the resulting set of bounding boxes representing detections from each test image, I then added to a heat map to represent the areas of high volumes of detection on the image.  Using a threshold to remove areas with only 1 detection was effective in removing the false-positive from our previous run across test images.

The heat maps were then finally used as input to `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap, representing complete vehicles.

I have included my output below showing the initial windowed detection, the resulting heatmap (after thresholding) and then the final detection area drawn on the same image.  The second row shows the thresholding technique effectively removing two single false-positives detected in the shadows which is great, but also shows how it can remove a correct identification in the second-last row of a vehicle that is further away.  This issue will be addressed in the video implementation section below

![alt text][image4]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Part 8 of the notebook has the pipeline for the video processing.  As already described above for the test images, I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Where the video implementation differs is in the need to track the positive detection locations across multiple frames to achieve a stable result, and eliminate false-positives flickering across one or two frames.  To do this, I defined `stored_box_list[]` to hold the bounding box values from the last 10 frames, and I would use all of these boxes to generate a single heat map.  I then increased the threshold to remove false positives to 5, meaning that only detections which strongly persisted across the majority of these 10 frames would be kept.

This resulted in a smoother bounding box output for the video, and prevented almost all noise from shadows and other markers in the video which the classifier may identify as a vehicle for a single frame or two.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I found a lot of false positives early on in the project, especially in the trees and sky strangely enough.  Limiting the area to deploy the window search to only the road removed these.

There are also some really tricky shadows compounded by some vehicles on the other side of the road around the 40 second mark of the video which improved a lot when averaging the heatmap over 10 frames and using a more aggressive thresholding level of 5.  This was a trade-off between ensuring adequate ongoing detections of the vehicles travelling in the same direction on the road, versus reducing these (quite persistent) false positives in that section.

The pipeline would likely struggle on other scenarios where there was a bigger mixture of oncoming traffic without a central divider, or vehicles moving rapidly across the video for only a few frames.  Vehicle types and obstacles not captured in the test set (more heavy vehicles, buses, cyclists etc) would also not be detected without more training data.

To make my pipeline more robust for a variety of road situations, I would try adding multiple windows passes at different scales to identify close vehicles and those further away more accurately.  It would also be worth experimenting with a deep learning approach to this problem, to avoid the time taken hand-crafting parameters or trying to crop road areas for detection in additional road scenarios (given sufficient images for training), given the success DL has had in image processing generally.

