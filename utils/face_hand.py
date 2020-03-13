
# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4  #Non-maximum suppression threshold

inpWidth = 416  #608     #Width of network's input image
inpHeight = 416 #608     #Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--dir', help='Path to image file.')
#parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()
print ("I AM HERE ")

def load_network(classesFile,cfg,weights):
    """
        classesFile - Name of all classes
        cfg         - Network Configuration
        weights     - Weights of network

    """
    # Load names of classes
    #classesFile = "classes.names"

    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # Give the configuration and weight files for the model and load the network using them.

    #modelConfiguration = "darknet-yolov3.cfg"
    #modelWeights = "backup/darknet-yolov3_354.weights"

    net = cv.dnn.readNetFromDarknet(cfg, weights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    return net , classes

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classes, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    #    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255), cv.FILLED)
    #cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs, classes,path,label=0 ):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        #print("out.shape : ", out.shape)
        for detection in out:
            #if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            #if scores[classId]>confThreshold:
            confidence = scores[classId]
            #if detection[4]>confThreshold:
                #print(detection[4], " - ", scores[classId], " - th : ", confThreshold)
                #print(detection)
            if confidence > confThreshold:
                index = path.rfind('/')
                destination = path[:index+1]
                imagename = path[index+1:].split('.')[0]
                f= open(destination+imagename+".json","a+") 
                f.write("\n")
                f.write(str(label)+" "+str(float(detection[0]))+" "+str(float(detection[1]))+" "+str(float (detection[2]))+" "+str(float(detection[2])))
                f.close()
                

                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classes, classIds[i], confidences[i], left, top, left + width, top + height)

# Arguments 
path = "darknet/"

classesFile_face = path + 'face/classes.names'
cfg_face = path + 'face/darknet-yolov3.cfg'
weights_face = path + 'face/weights/darknet-yolov3_1000.weights'

classesFile_hand = path + 'hand/classes.names'
cfg_hand = path + 'hand/darknet-yolov3.cfg'
weights_hand = path + 'hand/weights/darknet-yolov3_1100.weights'


# Load Networks
nets = []
nets.append( load_network(classesFile_face,cfg_face,weights_face) )
nets.append( load_network(classesFile_hand,cfg_hand,weights_hand) )



# Process inputs
winName = 'Deep learning object detection in OpenCV'
#cv.namedWindow(winName, cv.WINDOW_NORMAL)
listOfFiles = list()
for (dirpath, dirnames, filenames) in os.walk(args.dir):
  listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    #print ("Dir Path"+str(dirpath))
    #print ("dir names "+ str (dirnames[0]))
for image in listOfFiles :
    if os.path.exists(image) and (".jpg" in image):
        # Open the image file
        if not os.path.isfile(image):
            print("Input image file ", image, " doesn't exist")
            sys.exit(1)
        frame = cv.imread(image)
        outputFile = image[:-4]+'_yolo_out_py.jpg'
        print ("path to file: " + image)
        #sys.exit()
  
    #while cv.waitKey(1) < 0:
        
        # get frame from the video
        #hasFrame, frame = cap.read()
        
        # Stop the program if reached end of video
        #if not hasFrame:
            #print("Done processing !!!")
            #print("Output file is stored as ", outputFile)
            #cv.waitKey(3000)
            #break

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        for i in range(len(nets)):
            nets[i][0].setInput(blob)

        # Runs the forward pass to get output of the output layers
        for i in range(len(nets)):
            outs = nets[i][0].forward(getOutputsNames(nets[i][0]))

            # Remove the bounding boxes with low confidence
            if i == 0:
              postprocess(frame, outs, nets[i][1],image)
            elif i ==1 :
              postprocess (frame, outs, nets[i][1],image,1)
            # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
            t, _ = nets[i][0].getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
            #cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        index = image.rfind('/')
        destination = image[:index]
        imagename = image[index+1:].split('.')[0]
        if os.path.exists(destination+"/"+imagename+".json"):
          File= open(destination+"/"+imagename+".json","r")
          reading = File.readlines()
          facech = 0 
          handche= 0
          for r in reading :
            if (r!="\n"):
              if (r[0] == "0"):
                facech = 1 
              elif (r[0]=="1"):
                handche+=1 
          if (facech == 0 ):
              index = destination.rfind('/')
              no_detect = destination[:index+1]
              f=open (no_detect+"NoFaceDetection.txt","a+")
              f.write("\n")
              f.write(imagename+".jpg")
              f.close()
          if (handche <=1 ):
            if (handche ==1 ):
              index = destination.rfind('/')
              no_detect = destination[:index+1]
              f=open (no_detect+"OneHandDetection.txt","a+")
              f.write("\n")
              f.write(imagename+".jpg")
              f.close()
            else :
                index = destination.rfind('/')
                no_detect = destination[:index+1]
                f=open (no_detect+"NoHandDetection.txt","a+")
                f.write("\n")
                f.write(imagename+".jpg")
                f.close()
          File.close()
        else :
          index = destination.rfind('/')
          no_detect = destination[:index+1]
          f=open (no_detect+"NoDetection.txt","a+")
          f.write("\n")
          f.write(imagename+".jpg")
          f.close()
    #sys.exit()    
          
        #f.write("\n")
        #f.write(str(label)+" "+str(int(detection[0]))+" "+str(int(detection[1]))+" "+str(int (detection[2]))+" "+str(int(detection[2])))
                
        # Write the frame with the detection boxes
        #if (image):
        #cv.imwrite(outputFile, frame.astype(np.uint8))
        #sys.exit()
        #else:
            #vid_writer.write(frame.astype(np.uint8))

        # cv.imshow(winName, frame)