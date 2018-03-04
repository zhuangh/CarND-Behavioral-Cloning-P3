# *Self-Driving Car Behavioral Cloning Project*

by Hao Zhuang, 2018

##### The goals / steps of this project are the following:
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

##### Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  


##### My project includes the following files:
* model.ipynb contains the visualization of the data and the script to create and train the model (model.py is the same thing). (GPU is used for training of the neural networks in this project.)
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

##### Submission includes functional code

Using model.ipynb by Jupyter notebook. After running all the module, we will have the model saved in model.h5 and model.json.

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
Docker (CPU)
```sh
docker run -it --rm -p 4567:4567 -v `pwd`:/src udacity/carnd-term1-starter-kit python drive.py model.h5
```


### Model Architecture and Training Strategy


#### Final Model Architecture

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

#### Creation of the Training Set & Training Process

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

After the collection process, I had 96432 number of data points. I then preprocessed this data by Keras' fig_generate in Kera to save the GPU memory consumption.

I finally randomly shuffled the data set and put 10% of the data into a validation set.  I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20 from the figure below. I used an adam optimizer. Training Loss figure of the proposed model  in this work.

![alt text][nvidia_loss]

I also compare the propsed model to the pre-trained VGG model

![alt text][vgg_loss]

### Autonomous Driving Recording

[Autonomous Driving Recording 1](https://youtu.be/6ZO3cvuF3EY)

[Autonomous Driving Recording 2](https://youtu.be/87ycEDbTr1M)



### Summary and Future Directions

The neural network is based on Nvidia's End to End Learning for Self-Driving Cars https://arxiv.org/abs/1604.07316.

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer, so the learning rate was not tuned manually.

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. For details about how I created the training data, see the next section. 

The possible enhancement is listed as follows. 

- Translate the images horizontally and vertically to mimic driving uphill and downhill.

- Get training data from Track 2 and test the autonomous driving in the track 2.


