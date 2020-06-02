#  DeepAL

##  Deep Learning for Assisted Living



###  Description 

------

This project classifies human activities such as human falling and human choking in order to detect the problems which are most specifically faced by elderly people. The model proposed by this project uses the data which has been augmented by adding additional features introduced by the face, hand positions and keypoint models. We tackle the task component of increasing the accuracy of a model by training it on a small and structured dataset through data augmentation.




### Models

------

- Yolov3 (Darknet) for Face and Hand Detections

- AlphaPose for Key-point Estimations

- Resnet50 for classification of falling/non-falling and choking/non-choking




### Using the model

------

*demo.py* is the output file for testing the model on any of its weights. Testing works for image, video and webcam. When passed through this file, the results images are labelled at the top left of the image with the models predicted class i.e Falling or Non-Falling. The weights can be found [here](https://drive.google.com/open?id=1V3DrJsDEGXWdwETtMQY1hGtkkYwCNgIW)

#### <u>Testing</u>

- For image

`python demo.py --weightF <PathToWeightFall> --image <PathToImage> --aug ###`

- For video

`python demo.py --weightF <PathToWeightFall> --video <PathToVideo> --aug ###`

- For webcam

`python demo.py --weightF <PathToWeightFall> --aug ###`

#### <u>Arguments</u>

--weightF
Path to weight file of Falling model

--weightC
Path to weight file of Choking model

- --aug
2nd place bit - Face , 1st place bit - Hand , 0th place bit - Pose
Enable or disable the bit for model selection.

- --size
integer size for label averaging (default 5)

- --mirror
flip the webcam frame 

#### <u>Utils</u>

Few by-product utilities were also created such as:

- Detect only hands and face on a image can be done via utils/detectHandsFaces.ipynb
- Extracting frames from videos and storing them to given destination can be achieved via utils/ExtractFrames.ipynb




### Results

------



#### FALL

##### Plots Alpha Approach

- Raw
<img src="images/f_raw.png" width="450" height="300">

- Raw + Face + Hand
<img src="images/f_raw-face-hand.png" width="450" height="300">


##### Plots Beta Approach

- Raw + Face
<img src="images/f_raw-face.png" width="450" height="300">
- Raw + Hand
<img src="images/f_raw-hand.png" width="450" height="300">



#### Choke

##### Plots Alpha Approach

- Raw
<img src="images/c_raw.png" width="450" height="300">

- Raw + Face + Hand
<img src="images/c_raw-face-hand.png" width="450" height="300">


##### Plots Beta Approach

- Raw + Face
<img src="images/c_raw-face.png" width="450" height="300">
- Raw + Hand
<img src="images/c_raw-hand.png" width="450" height="300">


### Creators

------



<table>
    <td>
      <h3>
        Ahsan Abbas
  </h3>
        <img src="images/813.jpg" width="300" height="300">
    </td>

   <td>
  <h3>
        Muhammad Arslan
  </h3>
        <img src="images/IMG_5730_Frame84.jpg" width="300" height="300">
   </td>

   <td>
  <h3>
        Govinda Kumar
  </h3>
        <img src="images/VID2019120216265238.jpg" width="300" height="300">
    </td> 


​       
</table>


### Final Year Project

---

This is our final year project for Bachelors of Computer Science 2020.

**Supervisor:**
> Dr Omer Ishaq
> Dr Kashif Saghar

**Group Members:**

> Ahsan Abbas

> Muhammad Arslan 

> Govinda Kumar 
