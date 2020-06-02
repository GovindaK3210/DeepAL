
# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import torch
from collections import deque
from yolo_face_hand import load_network, execute_get_heatmap
import torchvision.transforms as transforms
#import thread

import sys
sys.path.insert(1,'./AlphaPose')

from AlphaPose.opt import opt
args = opt

#import threading
#import concurrent.futures
#from subprocess import check_output
import concurrent.futures


inpWidth = 224  #608     #Width of network's input image
inpHeight = 224 #608     #Height of network's input image



def draw_label(frame , label, model):
    if model == "Fall":
        if label == 0:
            lbl = 'Falling'
        else:
            lbl = 'Non Falling'
        backgroundStartPoint = (0,10)
        backgroundEndPoint = (200,50)
        textStartPosition = (10,40)
        
    else:
        if label == 0:
            lbl = 'Choking'
        else:
            lbl = 'Non Choking'
        height, width, channels = frame.shape
        backgroundStartPoint = (width-210,10)
        backgroundEndPoint = (width,50)
        textStartPosition = (width-200-5,40)

    font                   = cv.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = (0,128,0)
    lineType               = 2
    thickness              = -1

    # Draw white background rectangle
    frame = cv.rectangle(frame, backgroundStartPoint, backgroundEndPoint, (255,255,255), -1)

    # Draw label on white rectangle 
    cv.putText(frame,lbl, textStartPosition, font, fontScale,fontColor,lineType,thickness)

    return frame

"""
    frame   : original frame
    frame_  : augmented frame
    net     : network to pass frame
    Q       : queue
    model   : Fall/Choke
"""
def modelPrediction(frame,frame_,net, Q , model):
    with torch.set_grad_enabled(False):
        predictions = net(frame_)

    # Label Averaging 
    values, indices = predictions.max(1)
    Q.append(np.array(predictions.to('cpu')))

    results = np.array(Q).mean(axis=0)
    i = np.argmax(results)

    # Put label on frame
    return draw_label(frame,i, model)

#-----------------------------------------------------------------------------------------------
# Handle Arguments
#-----------------------------------------------------------------------------------------------

if len(args.aug) != 3:
    print("Invalid Augmentation..\nExiting..")
    exit(-1)

if args.weightF is None and args.weightC is None:
    print("No Model Selected..\nExiting..")
    exit(-1)


#-----------------------------------------------------------------------------------------------
# Handle Augmentation Models
#-----------------------------------------------------------------------------------------------

if args.aug[0] == '1':        # Load Face YOLO Model
    classesFile_face = 'Yolo-FaceHand/face/classes.names'
    cfg_face = 'Yolo-FaceHand/face/darknet-yolov3.cfg'
    weights_face = 'Yolo-FaceHand/face/darknet-yolov3_1000.weights'
    face_net, face_classes = load_network(classesFile_face,cfg_face,weights_face)
if args.aug[1] == '1':        # Load Hand YOLO Model
    classesFile_hand = 'Yolo-FaceHand/hand/classes.names'
    cfg_hand = 'Yolo-FaceHand/hand/darknet-yolov3.cfg'
    weights_hand = 'Yolo-FaceHand/hand/darknet-yolov3_1100.weights'
    hand_net , hand_classes= load_network(classesFile_hand,cfg_hand,weights_hand)
if args.aug[2] == '1':        # Load AlphaPose model
    from pose import *
    det_loader = DetectionLoader(None, batchSize=1) # YOLO Loaded for Human Detection
    sys.stdout.flush()
    # Load pose model
    pose_dataset = Mscoco()
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    pose_model.cuda()
    pose_model.eval()


#-----------------------------------------------------------------------------------------------
# Handle Fall/Choke Models
#-----------------------------------------------------------------------------------------------

transform = transforms.Compose([
                      #transforms.Resize((224,224)),
                      transforms.ToTensor()
                            ])


device = 'cuda'
# Load network
if args.weightF is not None:
    net_f = torch.load(args.weightF).to(device)
if args.weightC is not None:
    net_c = torch.load(args.weightC).to(device)

# Queue for label averaging
Q_f = deque(maxlen=int(args.size))
Q_c = deque(maxlen=int(args.size))
# Process inputs
winName = 'DeepAL Detection'
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
face_heatmap = hand_heatmap = pose_heatmap = None
while cv.waitKey(1) < 0:
    
    # get frame from the video
    hasFrame, frame = cap.read()
    
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        break

    frame_ = cv.resize(frame,(224,224))

    # TODO threading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        if args.aug[0] == '1':
            face_thread_return = executor.submit(execute_get_heatmap, face_net,face_classes,frame)
            #face_heatmap = thread_return.result()
            #face_heatmap = execute_get_heatmap(face_net,face_classes,frame)
            #face_heatmap = np.stack([ np.array(face_heatmap.resize((224,224))) ], axis=2)
        if args.aug[1] == '1':
            hand_thread_return = executor.submit(execute_get_heatmap, hand_net,hand_classes,frame)
            #hand_heatmap = execute_get_heatmap(hand_net,hand_classes,frame)
            #hand_heatmap = np.stack([ np.array(hand_heatmap.resize((224,224))) ], axis=2)
        if args.aug[2] == '1':
            data_loader = ModifiedImageLoader(frame, batchSize=1, format='yolo')
            data_loader.getitem_yolo()
            det_loader.dataloder = data_loader
            det_loader.update()
            det_processor = DetectionProcessor(det_loader)
            det_processor.update()
            pose_thread_return = executor.submit(execute_pose_heatmaps, det_processor,pose_model)
            #pose_heatmaps = execute_pose_heatmaps(det_processor,pose_model)
            #hm_pose = np.stack([ x.resize((224,224)) for x in pose_heatmaps ], axis=2)

        # This Needs to Be sequential
        if args.aug[0] == '1':
            face_heatmap = face_thread_return.result()
            face_heatmap = np.stack([ np.array(face_heatmap.resize((224,224))) ], axis=2)
            frame_ = np.concatenate((frame_,face_heatmap),axis=2)
        if args.aug[1] == '1':
            hand_heatmap = hand_thread_return.result()
            hand_heatmap = np.stack([ np.array(hand_heatmap.resize((224,224))) ], axis=2)
            frame_ = np.concatenate((frame_,hand_heatmap),axis=2)
        if args.aug[2] == '1':
            pose_heatmaps = pose_thread_return.result()
            hm_pose = np.stack([ x.resize((224,224)) for x in pose_heatmaps ], axis=2)
            frame_ = np.concatenate((frame_,hm_pose),axis=2)
    
    #frame_ = np.concatenate((frame_,face_heatmap,hand_heatmap,hm_pose),axis=2)
    print(frame_.shape)
    # TODO
    frame_ = np.float32(frame_)
    frame_ =  transform(frame_ / 255.0)
    frame_ = frame_.unsqueeze(0).to(device)

    # Create a 4D blob from a frame.
    #blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    #frames.append(frame)
    #frame_ = torch.from_numpy(frame).float().to(device)
    #frame_ = frame_.Resize((224,224))
    
    if args.weightF is not None:
        frame = modelPrediction(frame,frame_,net_f,Q_f,"Fall")
    if args.weightC is not None:
        frame = modelPrediction(frame,frame_,net_c,Q_c,"Choke")
      
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