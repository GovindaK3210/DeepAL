#  DeepAL

##  Deep Learning for Assisted Living



###  Description 

------

This project classifies human activities such as human falling and non-falling in order to detect the problems which are most specifically faced by elderly people. The model proposed by this project uses the data which has been augmented by adding additional features introduced by the face, hand positions and keypoint models. We tackle the task component of increasing the accuracy of a model by training it on a small and structured dataset through data augmentation.



### Models

------

- Yolov3 (Darknet) for Face and Hand Detections

- AlphaPose for Key-point Estimations

- Resnet50 for classification of falling and non-falling



### Using the model

------

*fallingDemo.py* is the output file for testing the model on any of its weights. Testing works for both image and video files and when passed through this file, the results images are labelled at the top left of the image with the models precited class i.e Falling or Non-Falling.

#### <u>Testing</u>

- For image

`python fallingDemo.py --weight <PathToWeight> --image <PathToImage>`

- For video

`python fallingDemo.py --weight <PathToWeight> --video <PathToVideo>`

#### <u>Utils</u>

Few by-product utilitiese were also created such as:

- Detect only hands and face on a image can be done via utils/detectHandsFaces.ipynb
- Extracting frames from videos and storing them to given destination can be achieved via utils/ExtractFrames.ipynb



### Results

------

#### Plots Alpha Approach

- Raw                          ![raw](images\raw.png)



- Raw + Face + Hand ![raw-face-hand](images\raw-face-hand.png)



#### Plots Beta Approach

- Raw + Face                ![raw-face](images\raw-face.png)

- Raw + Hand               ![raw-hand](images\raw-hand.png)





### Creators

------



**Ahsan Abbas**

![813](images\813.jpg)



**Muhammad Arslan**

![IMG_5730_Frame84](images\IMG_5730_Frame84.jpg)



**Govinda Kumar**

![VID2019120216265238](images\VID2019120216265238.jpg)



### Final Year Project

---

This is our final year project for Bachelors of Computer Science 2020.

**Group Members:							Supervisor:**

> Ahsan Abbas									Dr Omer Ishaq

> Muhammad Arslan 

> Govinda Kumar 
