# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[brake_dist]: ./images/brake.png "Brake"
[throttle_dist]: ./images/throttle.png
[steering_dist]: ./images/steering.png
[flip_steering]: ./images/flip_steering_image.png
[speed_dist]: ./images/speed.png
[straight]: ./images/straight.png
[straight_crop]: ./images/straight_crop.png
[left]: ./images/left.png
[left_crop]: ./images/left_crop.png
[right]: ./images/right.png
[right_crop]: ./images/right.png
[brightness1]: ./images/brightness_1.png
[brightness2]: ./images/brightness_2.png
[brightness3]: ./images/brightness_3.png
[vgg_loss]: ./images/vgg_loss.png
[nvidia_loss]: ./images/nvidia_loss.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.ipynb contains the visualization of the data and the script to create and train the model (model.py is the same thing)
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
Docker (CPU)
```sh
docker run -it --rm -p 4567:4567 -v `pwd`:/src udacity/carnd-term1-starter-kit python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py and model.ipynb). Here is the configuration of the architecture.

| Layer (type) |     Output Shape  |        Param #  |   Connected to                 |
|--------------------------------|------------------|-------:|-----------------|
|lambda_1 (Lambda)              |  (None, 49, 224, 3)   | 0 |          lambda_input_1[0][0]  |      
|convolution2d_1 (Convolution2D) | (None, 49, 224, 24) |  1824    |    lambda_1[0][0]   |    |      
|maxpooling2d_1 (MaxPooling2D) |   (None, 24, 112, 24)  | 0     |      convolution2d_1[0][0]|       
|dropout_1 (Dropout)   |           (None, 24, 112, 24) |  0       |    maxpooling2d_1[0][0] |            
|convolution2d_2 (Convolution2D) | (None, 24, 112, 36)  | 21636   |    dropout_1[0][0]       |           
|maxpooling2d_2 (MaxPooling2D)    |(None, 12, 56, 36)  |  0       |    convolution2d_2[0][0]  |          
|dropout_2 (Dropout)             | (None, 12, 56, 36)  |  0        |   maxpooling2d_2[0][0]   |          
|convolution2d_3 (Convolution2D) | (None, 12, 56, 64)  |  57664    |   dropout_2[0][0]       |           
|maxpooling2d_3 (MaxPooling2D)  |  (None, 6, 28, 64)  |   0        |   convolution2d_3[0][0]  |          
|dropout_3 (Dropout)           |   (None, 6, 28, 64)  |   0          | maxpooling2d_3[0][0]   |          
|convolution2d_4 (Convolution2D) | (None, 6, 28, 64)  |   36928    |   dropout_3[0][0]   |              
|maxpooling2d_4 (MaxPooling2D)  |  (None, 3, 14, 64)  |   0           convolution2d_4[0][0]   |         
|flatten_1 (Flatten)           |   (None, 2688)    |      0        |   maxpooling2d_4[0][0]   |          
|dropout_4 (Dropout)          |    (None, 2688)   |       0        |   flatten_1[0][0]      |            
|dense_1 (Dense)            |      (None, 1164)   |       3129996 |    dropout_4[0][0]    |              
|dropout_5 (Dropout)         |     (None, 1164)   |       0         |  dense_1[0][0]      |              
|dense_2 (Dense)              |    (None, 512)     |      596480    |  dropout_5[0][0]    |              
|dropout_6 (Dropout)          |    (None, 512)   |        0         |  dense_2[0][0]   |                 
|dense_3 (Dense)              |    (None, 50)   |         25650    |   dropout_6[0][0]   |               
|dropout_7 (Dropout)        |      (None, 50)    |        0       |    dense_3[0][0]      |              
|dense_4 (Dense)            |      (None, 10)       |     510       |  dropout_7[0][0]    |             
|dropout_8 (Dropout)        |      (None, 10)    |        0     |      dense_4[0][0]     |           
|dense_5 (Dense)    |              (None, 1)        |     11      |    dropout_8[0][0]     |            

Total params: 3,870,699
Trainable params: 3,870,699
Non-trainable params: 0

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

##### "Brake" Distribution 
![alt text][brake_dist]

##### "Throttle" Distribution 
![alt text][throttle_dist]

##### "Steering" Distribution 
![alt text][steering_dist]

##### "Speed" Distribution 
![alt text][speed_dist]


##### "Striaght Running" Snapshot
![alt text][straight]

##### Cropped "Striaght Running" Snapshot
![alt text][straight_crop]
![alt text][brightness]
##### "Left Turn" Snapshot
![alt text][left]

##### Cropped "Left Turn" Snapshot
![alt text][left_crop]

##### "Right Turn" Snapshot
![alt text][right]

##### Cropped "Right Turn" Snapshot
![alt text][right_crop]

To augment the data set, I also flipped images and angles thinking that this would help with the left turn bias involves flipping images and take the opposite sign of the steering measurement.

![alt text][steering_dist]

After flipping, the distribution looks more balanced. 

![alt text][flip_steering]

##### Brightness Augmentation
![alt text][brightness1]
![alt text][brightness2]
![alt text][brightness3]


After the collection process, I had 96432 number of data points. I then preprocessed this data by fig_generate in Kera to save the GPU memory consumption.

I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 

The ideal number of epochs was 20 from the figure below. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Training Loss figure of the proposed model  in this work.
![alt text][nvidia_loss]

I also compare the propsed model to the pre-trained VGG model

![alt text][vgg_loss]

### Autonomous Driving Recording
[Autonomous Driving Recording](https://youtu.be/6ZO3cvuF3EY)
[Autonomous Driving Recording](https://youtu.be/87ycEDbTr1M)

