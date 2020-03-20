
# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import torch
from collections import deque

inpWidth = 224  #608     #Width of network's input image
inpHeight = 224 #608     #Height of network's input image

parser = argparse.ArgumentParser(description='Fall Detection using Resnet in OPENCV')
parser.add_argument('--weight', help='Path to weight file.')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')

parser.add_argument('--size', help='size of frames')
parser.add_argument('--mirror', default=True,help='Mirror webcam')

args = parser.parse_args()

print ("I AM HERE ")

def load_network(weights):
    """
        classesFile - Name of all classes
        cfg         - Network Configuration
        weights     - Weights of network

    """

    net = torch.load(weights).to(device)

    return net 


def draw_label(frame , label):
    if label == 0:
        lbl = 'Falling'
        endpoint = (320,150)
    else:
        lbl = 'Non Falling'
        endpoint = (550,150)

    font                   = cv.FONT_HERSHEY_SIMPLEX
    topLeftCornerOfText    = (20,100)
    fontScale              = 3
    fontColor              = (0,128,0)
    lineType               = 2
    thickness              = -1

    # Draw white background rectangle
    frame = cv.rectangle(frame, (10,10), endpoint, (255,255,255), -1)

    # Draw label on white rectangle 
    cv.putText(frame,lbl, topLeftCornerOfText, font, fontScale,fontColor,lineType,thickness)

    return frame


device = 'cuda'
# Load network
net = load_network(args.weight).to(device)
Q = deque(maxlen=int(args.size))

# Process inputs
winName = 'DeepAL Fall Detection'
#cv.namedWindow(winName, cv.WINDOW_NORMAL)
outputFile = "resnet_out_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_resnet_out_py.jpg'
    print ("path to file: " + args.image[:-4])
    #sys.exit()
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_resnet_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
if (args.video):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

#epoch = 45
#device = 'cuda'   
#model = torch.load('weights/alpha/raw/epoch_%d.pth'%(epoch)).to(device)
frames=[]
labels = []
count = 0 
while cv.waitKey(1) < 0:
    
    # get frame from the video
    hasFrame, frame = cap.read()
    
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        break

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    frames.append(frame)
    frame_ = torch.from_numpy(blob).float().to(device)
    #frame_ = frame_.Resize((224,224))
    
    # Pass the frame through network
    with torch.set_grad_enabled(False):
        predictions = net(frame_)
    
    # Label Averaging 
    values, indices = predictions.max(1)
    Q.append(np.array(predictions.to('cpu')))

    results = np.array(Q).mean(axis=0)
    i = np.argmax(results)

    # Put label on frame
    draw_label(frame,i)
    
    # Output
    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8))
    elif(args.video):
        vid_writer.write(frame.astype(np.uint8))
    else:
        if( args.mirror ):
            frame = cv.flip(frame,1)
        
        cv.imshow(winName, frame)
        if cv.waitKey(1) == 27:
            break

cv.destroyAllWindows()