# Action-recognition-in-Still-Images
In this <b><a href = "https://github.com/shin71/Action-recognition-in-Still-Images/blob/main/Recognizing_actions_in_still_images_distracted_driver_detection.ipynb">notebook</a></b> I show how to recognize actions in still images well using Transfer Learning and Attention Mechanism

## Link for the model trained 
https://drive.google.com/file/d/1O2gzlCs5id1MZQbCjp1DSWfhuso_wzUf/view?usp=sharing
You can load this model directly as demonstrated in the notebook to start making predictions as soon as possible
<h3>Input requirments of the model</h3>
<ul>
  <li><b> The input has to be of the shape 64*86*3</b></li>
  <li><b> Input needs to preprocessed using the https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/preprocess_input<b></li>
</ul>
    
## Link for the problem 
<b><a href="https://www.kaggle.com/c/state-farm-distracted-driver-detection">State Farm Distracted Driver Detection</a></b>
    
# Distracted Driver Detection

## Using Computer vision to classify the behavior of the driver
```
    By-
Shivansh Singla
UIET
```
    
# Explaining the problem

<span style="color:#6AA84F"> We are given driver images\, each taken in a car with a driver doing something in the car \(texting\, eating\, talking on the phone\, makeup\, reaching behind\, etc\)\.</span>

<span style="color:#6AA84F">  Our goal is to predict the likelihood of what the driver is doing in each picture\.</span>

<span style="color:#6AA84F"> We will be given 10 different classes in which we can classify the  </span>  <span style="color:#6AA84F"> activity being done by  the  driver\. </span>  <span style="color:#6AA84F"> </span>

![](img/Learnings%20-%20Recognizing%20actions%20in%20still%20images%20and%20analysis%20the%20data%20in%20the%20problem0.png)

# Dataset description

![](img/Learnings%20-%20Recognizing%20actions%20in%20still%20images%20and%20analysis%20the%20data%20in%20the%20problem1.png)

* Our dataset will have one folder and one csv file
* Folder called ‘imgs’ contain 2 directories \- train and test\.
  * Train contains 10 directories each labelled by class i\.e c0\,c1\,c2\,\.\.\. where each directory contains the images of the class given by directory label
  * Total images in train set are 22424 where each class contributes approximately same number of images
  * Test folder contains approx 80000 unlabelled images on which we have to make predictions
* The csv is driver\_imgs\_list\.csv It contains data about train images like which driver\_id is corresponding to which image as there are many photos of the same driver in the dataset

# Grading and Submission details

As the predictions will be graded on kaggle we need to make our predictions submittable in a csv format which are in the format defined by kaggle\.

Submissions are evaluated using the multi\-class logarithmic loss which is also called  <span style="color:#0000FF"> Categorical </span>  <span style="color:#0000FF"> cross entropy </span> \. Each image has been labeled with one true class\.

For each image\, you must submit a set of predicted probabilities \(one for every image\)\.

<span style="color:#00FFFF">Function for grading</span>

![](img/Learnings%20-%20Recognizing%20actions%20in%20still%20images%20and%20analysis%20the%20data%20in%20the%20problem3.png)

<span style="color:#00FFFF">Submission format \-></span>

![](img/Learnings%20-%20Recognizing%20actions%20in%20still%20images%20and%20analysis%20the%20data%20in%20the%20problem2.png)
    
# How to download the dataset

We will be using Kaggle API to download the dataset

Install kaggle package in python using  <span style="color:#00FFFF">\!pip install kaggle</span> \.You can check if its already present using  <span style="color:#00FFFF">\!pip show kaggle</span> \.

Upload the kaggle\.json file which contains the api key to root directory  ```~/\.kaggle/kaggle\.json```

Use  ```!kaggle competitions download -c state-farm-distracted-driver-detection``` to download it in zip format

# Loading/Analysing/Cleaning the data

* The dataset is in zip folder so first we need to unzip the dataset which can simply be done by
  * <span style="color:#0000FF">\!unzip /content/brain\-tumor\-dataset\-with\-saliency\.zip</span>
* So now we will load the driver\_imgs\_list\.csv using pandas and describe it
  * <span style="color:#000000">driver\_index = pd\.read\_csv\(</span>  <span style="color:#A31515">'driver\_imgs\_list\.csv'</span>  <span style="color:#000000">\)</span>
  * <span style="color:#795E26">print</span>  <span style="color:#000000">\(driver\_index\.describe\(\)\,end = </span>  <span style="color:#A31515">'\\n\\n'</span>  <span style="color:#000000">\)</span>
* We get to know that there are only 26 drivers which contribute to total 22424 images
* We also realize most of the classes have nearly equal number of images and each subject also has classes divided nearly equally b/w their actions
* So our model won’t be biased towards a certain class
* We will pick the train\(X\_train1\,Y\_train1\) and validation test\(X\_train2\,Y\_train2\) subjects alternatively starting from the most frequently occurring subject to the least occurring subject
* Reasons for using this instead of simple train\_test\_split will be explained in later slides
* We get names of every image belonging to each unique \(subject\,class\) pair using two loops and dataframe slicing
* After preprocessing input\(reason later\)\,We load the image and class to
  * X\_train1\,Y\_train1 if index is even
  * X\_train2\,Y\_train2 if index is odd
* All the subjects are sorted by number of images belonging to them
* We convert these lists to np array because most functions of tensorflow require np arrays
* We one hot encode the outputs

# Image preprocessing

* Initially sizes of images is 480\*640\*3 which we change to 64\*86\*3\(maintaining the aspect ratio\) because images are so much in number that they can’t be loaded in high resolution\. I even had to make predictions in running instead of making an X\_test then predicting at once because testing data has nearly 80000 images\.
* We also preprocess inputs using
  * <span style="color:#AF00DB">from</span>  <span style="color:#000000"> tensorflow\.keras\.applications\.resnet </span>  <span style="color:#AF00DB">import</span>  <span style="color:#000000"> preprocess\_input</span>
  * Because Pretrained models have specific input conditions which need to be fulfilled for good results
* The model also has a cropping layer to crop some part of image which doesn’t contribute anything to the activity which we are trying to detect

# Reason for the specific type of splitting

* A big challenge in action recognition in still images is the lack of large enough datasets\, which is problematic for training deep Convolutional Neural Networks \(CNNs\) due to the overfitting issue\.
* A simple train test split in proportion will just overfit the model and validation scores will also be good as subjects are same in both training and validation set\.Also\,because
  * Instead of Learning the activity being performed by the driver
  * It learns If a specific driver is doing this then activity being done is\. So In Predictions it looks for cues similar to a particular set\(train\) of drivers performing actions instead of actions themselves
* The only reason to crop was to focus more on the activity
* THIS WAS A VERY IMPORTANT THING I LEARNED
* Most of the notebooks you find on kaggle just simply use train test split to get fake impressive validation scores and never show their submission score
* When i started out I was also getting amazing validation scores and bad submission scores took me a long time to figure out the reason because every notebook contained False results\. I only realized it after analysing the data

# Some improvements in data that can be made to improve training Process

If Ram available is sufficient enough we can resize the image to higher resolutions like 256\*256\*3 which will help in better training

We can directly save the images in memory then predict at once instead of Running\.The former method I found out is at least 7 to 8 times faster when i was working with smaller images\(32\*32\*3\) for testing the model\.

Data Augmentation can be applied for better training and preventing overfitting to some extent

Hyper parameter tuning can also be applied

Ensemble learning is also possible

THIS MADE ME REALIZE GOOD RESULTS START AT DATA MANIPULATION AND NOT JUST AT THE MODEL ARCHITECTURE

# Transfer Learning

I will be using Resnet50 model as the pretrained model without its top layer which is used for classification to get some information about the images on which we work

Output of Resnet50 without top layer is Height \* channels \* width i\.e data format is “channels last”

We have kept the pooling as none because that will be done by the CBAM block

Resnet50 model from keras need to have a specific input format which can be obtained by preprocessing images with help of the module  _[https://www\.tensorflow\.org/api\_docs/python/tf/keras/applications/resnet50/preprocess\_input](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/preprocess_input)_

We have to make this layer not trainable

# Convolutional Block attention Module Architecture

CBAM is an attention module which is used to make CNN learn and focus more on the important information\, rather than learning non\-useful background information\. In the case of object detection\, useful information is the objects or target class crop that we want to classify and localize in an image\.

_[https://medium\.com/visionwizard/understanding\-attention\-modules\-cbam\-and\-bam\-a\-quick\-read\-ca8678d1c671](https://medium.com/visionwizard/understanding-attention-modules-cbam-and-bam-a-quick-read-ca8678d1c671)_  for explanation

![](img/Learnings%20-%20Recognizing%20actions%20in%20still%20images%20and%20analysis%20the%20data%20in%20the%20problem4.png)

![](img/Learnings%20-%20Recognizing%20actions%20in%20still%20images%20and%20analysis%20the%20data%20in%20the%20problem5.png)

![](img/Learnings%20-%20Recognizing%20actions%20in%20still%20images%20and%20analysis%20the%20data%20in%20the%20problem6.png)

# Final Neural network
     note - at the end of model there are 10 neurons not 40 as problem has 10 classes
![](img/Learnings%20-%20Recognizing%20actions%20in%20still%20images%20and%20analysis%20the%20data%20in%20the%20problem7.png)

# SUBMISSION SCORES

The best score and the latest score is the one which was obtained with Transfer Learning \+ Attention Mechanism

The second most recent score was the one obtained when pretrained model was trained from scratch\(option 2 model in the notebook\)

The score 30 was a blunder I forgot to preprocess input

All the other ones were some brute force which i was trying on transfer learning by changing image sizes and epochs\,etc\.

![](img/Learnings%20-%20Recognizing%20actions%20in%20still%20images%20and%20analysis%20the%20data%20in%20the%20problem8.png)


