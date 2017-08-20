# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

## Histogram of Oriented Gradients (HOG)

### Explain how (and identify where in your code) you extracted HOG features from the training images.

All the code is included in the 

In order to automate the process of looking for the features to extract, I created an sklearn `FeatureUnion` where I plugged several custom sklearn transformations in order to extract the different features. This way, I can automate the search of the params for the features extractors with `GridSearchCV` or `RandomSearchCV`. 

This is the pipeline configuration:

``` python
svc_pipeline = Pipeline([
        ('color', ColorConverter()),
        ('feat', FeatureUnion([
            ('bin_spatial', SpatialBins()),
            ('color_hist', ColorHistogram()),
            ('hog', HOG())
        ])),
        ('scl', StandardScaler()),
        # ('pca', PCA()),
        ('clf', LinearSVC())
    ],
    memory='./param_search_cache/'
)
```

And this is the implementation of the Feature extractors:

``` python
def bin_spatial(img, size=(32, 32)):
    """ Compute binned color features"""

    features = cv2.resize(img, size).ravel() 
    return features

def color_hist(img, nbins=32, bins_range=(0, 256)):
    """ Compute color histogram features"""
    
    channels_hist = [np.histogram(img[:,:,c], bins=nbins, range=bins_range)[0] for c in range(img.shape[-1])]
    hist_features = np.concatenate(channels_hist)
    return hist_features


def convert_color(image, cspace='RGB'):
    """ Convert image to color space """
    
    mapping = {
        'HSV':  cv2.COLOR_RGB2HSV,
        'LUV':  cv2.COLOR_RGB2LUV,
        'HLS':  cv2.COLOR_RGB2HLS,
        'YUV':  cv2.COLOR_RGB2YUV,
        'YCrCb': cv2.COLOR_RGB2YCrCb,
    }
        
    if cspace != 'RGB':
        return cv2.cvtColor(image, mapping[cspace])
    else: 
        return image


def get_hog_features(img, orient, pixels_per_cell, cells_per_block, transform_sqrt=True, vis=False, feature_vec=True):    
    """ Returns HOG features and visualization """
    
    return hog(img, orientations=orient, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
               cells_per_block=(cells_per_block, cells_per_block), transform_sqrt=transform_sqrt, 
               visualise=vis, feature_vector=feature_vec)

def hog_features(img, orient, pixels_per_cell, cells_per_block, transform_sqrt=True, channels=[0]):    
    feats = []
    for channel in channels:
        feats.append(get_hog_features(img[:,:,channel], 
                            orient, pixels_per_cell, cells_per_block))
    return np.ravel(feats)   

class ColorConverter(BaseEstimator, TransformerMixin):
    """ Feature extraction wrapped in a sklearn transformer. """

    def __init__(self, cspace='RGB'):
        self.cspace=cspace
        
    def fit(self, x, y=None, **fit_params):
        return self
    
    def transform(self, X):
        images = []
        for image in X:
            images.append(convert_color(image, self.cspace))
        return images
            
class SpatialBins(BaseEstimator, TransformerMixin):
    """ Feature extraction wrapped in a sklearn transformer. """

    def __init__(self, spatial_size=(32, 3)):
        self.spatial_size=spatial_size      

    def fit(self, x, y=None, **fit_params):
        return self
    
    def transform(self, X):
        images = []
        for image in X:
            images.append(bin_spatial(image, self.spatial_size))
        return images
    
class ColorHistogram(BaseEstimator, TransformerMixin):
    """ Feature extraction wrapped in a sklearn transformer. """

    def __init__(self, hist_bins=32, hist_range=(0, 256)):
        self.hist_bins=hist_bins
        self.hist_range=hist_range

    def fit(self, x, y=None, **fit_params):
        return self
    
    def transform(self, X):
        images = []
        for image in X:
            images.append(color_hist(image, self.hist_bins, self.hist_range))
        return images

class HOG(BaseEstimator, TransformerMixin):
    """ Feature extraction wrapped in a sklearn transformer. """

    def __init__(self, orient=12, pixels_per_cell=8, cells_per_block=2, transform_sqrt=True, channels=[0]):
        self.orient=orient
        self.pixels_per_cell=pixels_per_cell
        self.cells_per_block=cells_per_block
        self.channels=channels
        self.transform_sqrt=transform_sqrt

    def fit(self, x, y=None, **fit_params):
        return self
    
    def transform(self, X):
        images = []
        for image in X:
            images.append(hog_features(image, 
                           orient=self.orient, 
                           pixels_per_cell=self.pixels_per_cell, 
                           cells_per_block=self.cells_per_block,
                           channels=self.channels,
                           transform_sqrt=self.transform_sqrt))
        return images
```

In the first versions I also applied a PCA step in order to reduce the features vector dimension. At the end, I decided to remove this step, as I had some memory issues (I even exahusted the swap partition in my computer!) with parallel cross validation using up to 16 threads and it didn't improve substantially the speed nor the accuracy of the model.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

