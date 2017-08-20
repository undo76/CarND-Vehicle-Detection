# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 

[//]: # (Image References)
[samples]: ./output_images/samples.jpg
[windows_1]: ./output_images/windows_1.jpg
[windows_2]: ./output_images/windows_2.jpg

[video]: ./project-video-result.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

All the code is included in the [Jupyter notebook](./Vehicle-detection.ipynb)

## Feature selection

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

    def __init__(self, spatial_size=(32, 32)):
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

In preliminary versions, I applied a PCA step in order to reduce the features vector dimension. At the end, I decided to remove this step, as I had some memory issues (I even exahusted the swap partition in my computer!) with parallel cross validation using up to 16 threads and it didn't improve substantially the speed nor the accuracy of the model.

###  Explain how you settled on your final choice of HOG parameters.

In order to select the parameters of the classifier and the features to extract, I configured the range of the parameters to explore.

``` python
param_dist = {
    'color__cspace': ['RGB', 'LUV', 'HLS', 'YUV', 'YCrCb'],
    'feat__bin_spatial__spatial_size': [(32, 32), (16, 16)],
    'feat__color_hist__hist_bins': [32, 16],    
    'feat__hog__channels': [(0,)],
    'feat__hog__orient': [8, 12],
    'feat__hog__pixels_per_cell': [8, 16],
    'feat__hog__cells_per_block': [2, 4],
#     'pca__n_components': [None, .9, .93, .95, .97, .99],
#     'pca__whiten': [ True, False ],
    'clf__C': np.logspace(-4, 2, 7), # [0.0001, ..., 100]
```

Then I used a `RandomizedSearchCV` instance using 3-folds to cross-validate 200 random models.

``` python
param_search = RandomizedSearchCV(
    estimator=svc_pipeline, 
    param_distributions=param_dist,
    scoring='accuracy',
    cv=3,
    n_iter=200,
    n_jobs=12,
    verbose=10
)
```

### Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

#### Data preparation

I used the dataset provided from Udacity. These example images come from a combination of the GTI vehicle image database, the KITTI vision benchmark suite, and examples extracted from the project video itself.

![Samples][samples]

Then I kept aside a 20% of the samples for validation. Unfortunately, the data is extracted from video, it means that the same car appears in several samples. Therefore we are leaking some information of the validation set into the training set. Solving this issue, would require to inspect manually all the samples and partition them in a way that the same car doesn't appear in both datasets.

``` python
# Combine datasets (set y = 1 if is a vehicle; 0 otherwise)
X = np.vstack([vehicles, non_vehicles])
y = np.hstack([np.ones(len(vehicles)), np.zeros(len(non_vehicles))])

# Shuffles and splits the data in a stratified fashion
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y)
print('Shapes: X_train={} X_test={}'.format(X_train.shape, X_test.shape))
print(np.max(X))
``` 
#### Training and model selection

After some exploration, I decided to use a Linear SVM. I also played with RBF kernels, but the training was very slow and memory intensive. Intuitively, I think that a neural network would work well, but as a challenge, I wanted to see how far a simple linear model could go with a good feature selection. 

As described in the previous section, I built a classification pipeline and I trained 200 random combinations of the hyperparameters. 

``` python
clf = param_search.best_estimator_
%time clf.score(X_test, y_test)
```

The best classifier gives an accuracy of over 99% on the test set. Nevertheless, we should expect less accuracy in real examples as some information was leaked into the training set because the dataset wasn't partitioned properly in order to prevent the same car from appearing in the training and the test set.

```
CPU times: user 5.21 s, sys: 23.9 ms, total: 5.24 s
Wall time: 5.36 s
0.99042792792792789
```

I inspected the results of the model selection and persisted the best classifier. 

``` python
clf.get_params()
```

Gives the selection of hyperparameters. (some lines omitted, for brevity):

``` python
{'clf': LinearSVC(C=0.001, class_weight=None, dual=True, fit_intercept=True,
      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
      multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
      verbose=0),
 
 ...
 
 'color': ColorConverter(cspace='YUV'),
 
 ...
 
 'feat': FeatureUnion(n_jobs=1, transformer_list=[
    ('bin_spatial', SpatialBins(spatial_size=(16, 16))), 
    ('color_hist', ColorHistogram(hist_bins=32, hist_range=(0, 256))), 
    ('hog', HOG(cells_per_block=2, channels=(0,), orient=12, pixels_per_cell=16, transform_sqrt=True))], transformer_weights=None),
  
  ...
}
```

## Sliding Window Search

In order to feed the classifier I have created a method to apply the sliding windows method. The size, the position and the steps are configurable parameters. I take into account the position of the horizon and the ratio of the window size that lies above and below the horizon to calculate the position (it is invariant to perspective projection). 


``` python
def find_cars(frame, 
              horizon_y = 420,
              size = 64,
              position_ratio = 5, # position_ratio = (bottom - horizon) / (horizon - top)
              min_window_size = 64,
              max_window_size = 280,
              n_window_sizes = 5,
              step_relative = 3,
              draw_findings = True,
              draw_windows = False
             ):

    step = size // step_relative
    window_sizes = np.linspace(min_window_size, max_window_size, n_window_sizes).astype(np.int)
    tops = horizon_y - (window_sizes) // position_ratio 
    
    windows = []
    positions = []
    for window_size, y1 in zip(window_sizes, tops):
        y2 = y1 + window_size
        scale = window_size / size
        region_width = frame.shape[1] * size // window_size
        region = cv2.resize(frame[y1:y2,...], (region_width, size))
        x_margin = (region_width % step) // 2 
        for x1_r in range(x_margin, x_margin + region_width - size, step):
            x2_r = x1_r + size 
            x1, x2 = int(x1_r * scale), int(x2_r * scale)
            windows += [region[:, x1_r:x2_r]]
            positions += [(x1, y1, x2, y2)]
        
    predictions = classifier.predict(windows)
    
    copy = None
    heatmap = np.zeros(frame.shape[:-1])
    
    if draw_windows or draw_findings:
        copy = np.zeros_like(frame)
        
    for pred, (x1, y1, x2, y2) in zip(predictions, positions):
        if draw_windows:
            cv2.rectangle(copy, (x1,y1), (x2,y2), (255,255,0), 3)
        if pred == 1:
            heatmap[y1:y2, x1:x2] += 1 
            if draw_findings:
                cv2.rectangle(copy, (x1,y1), (x2,y2), (0,0,255), 3)
            
    return heatmap, copy
```

### Examples

Result with `n_window_sizes = 3` and `step_relative = 1`

![Sliding Windows 1][windows_1]

Result with the default parameters (used in the video)

![Sliding Windows 1][windows_2]

### Video Implementation

Here's a [link to my video result](./project-video-result.mp4)

#### False positives filtering

In order to remove false positives I applied a temporal low-pass filter (smoothing) over the heatmap of detections. In order to do so I apply smooth factor to the previous heatmap (0 = no influence, 1 = don't forget). I found that values around 0.9 work the best. I also applied a threshold to the heatmap to filter out false positives. Then I segment the heatmap and calculate the bounding box of the thresholded blobs, using skimage's `label()`.

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I am pretty satisfied with the solution. 


Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

