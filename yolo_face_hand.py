import cv2
import numpy as np
from PIL import Image,ImageDraw
import copy

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4  #Non-maximum suppression threshold

inpWidth = 416  #608     #Width of network's input image
inpHeight = 416 #608     #Height of network's input image



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

    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return net , classes

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box YOLO
def drawPred(frame ,classes, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    #    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255), cv2.FILLED)
    #cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),    (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)


# Remove the bounding boxes with low confidence using non-maxima suppression NMS , Threshold
def postprocess(frame, outs, classes):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    print("w:",frameWidth," h:",frameHeight)

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        print("out.shape : ", out.shape)
        for detection in out:
            #if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            #if scores[classId]>confThreshold:
            confidence = scores[classId]
            if detection[4]>confThreshold:
                print(detection[4], " - ", scores[classId], " - th : ", confThreshold)
                print(detection)
            if confidence > confThreshold:
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
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    
    
    
    p = copy.deepcopy(frame)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(p ,classes, classIds[i], confidences[i], left, top, left + width, top + height)
    cv2.imwrite("test/pred.jpg",p)

    return boxes 
    
# YOLO 
def IoU(box1, box2):
    """
    calculate intersection over union cover percent
    :param box1: box1 with shape (N,4) or (N,2,2) or (2,2) or (4,). first shape is preferred
    :param box2: box2 with shape (N,4) or (N,2,2) or (2,2) or (4,). first shape is preferred
    :return: IoU ratio if intersect, else 0
    """
    # first unify all boxes to shape (N,4)
    if box1.shape[-1] == 2 or len(box1.shape) == 1:
        box1 = box1.reshape(1, 4) if len(box1.shape) <= 2 else box1.reshape(box1.shape[0], 4)
    if box2.shape[-1] == 2 or len(box2.shape) == 1:
        box2 = box2.reshape(1, 4) if len(box2.shape) <= 2 else box2.reshape(box2.shape[0], 4)
    point_num = max(box1.shape[0], box2.shape[0])
    b1p1, b1p2, b2p1, b2p2 = box1[:, :2], box1[:, 2:], box2[:, :2], box2[:, 2:]

    # mask that eliminates non-intersecting matrices
    base_mat = np.ones(shape=(point_num,))
    base_mat *= np.all(np.greater(b1p2 - b2p1, 0), axis=1)
    base_mat *= np.all(np.greater(b2p2 - b1p1, 0), axis=1)

    # I area
    intersect_area = np.prod(np.minimum(b2p2, b1p2) - np.maximum(b1p1, b2p1), axis=1)
    # U area
    union_area = np.prod(b1p2 - b1p1, axis=1) + np.prod(b2p2 - b2p1, axis=1) - intersect_area
    # IoU
    intersect_ratio = intersect_area / union_area

    return base_mat * intersect_ratio


# boxes .. YOLO format -> # TODO recheck
def get_heatmap(boxes, img_w , img_h):      # GENERATE HEATMAP
        
        hm = Image.new('L', (img_h, img_w))
        if boxes is not None  :
            for box in boxes :
              
              w = box[2]
              h = box[3]
              cx = box[0]+(w/2)
              cy = box[1]+(h/2)
              
              xmin =  (box[0])
              ymin = (box[1])
              xmax = (cx + (w/2))
              ymax = (cy + (h/2))
              hmDraw = ImageDraw.Draw(hm)                   
              hmDraw.rectangle([xmin,ymin,xmax,ymax], fill ="#ffffff")
        return hm

# blob - 4d blob from a frame.
def execute_get_heatmap(net,classes,orig_img):
    # Create 4d blob
    blob = cv2.dnn.blobFromImage(orig_img, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Forward Pass
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    boxes = postprocess(orig_img, outs, classes)

    # Generate Heatmap
    frame = get_heatmap(boxes,orig_img.shape[0],orig_img.shape[1])
    
    return frame
    
    
    #frame.save("test/frame_hand_hm_2_hw.jpg")
    #cv2.imwrite("test/frame_face.jpg",frame)
    #print("Written")


#def detect_and_generate_heatmap(net,classes,orig_img):

    #classesFile_face = 'aug/face/classes.names'
    #cfg_face = 'aug/face/darknet-yolov3.cfg'
    #weights_face = 'aug/face/darknet-yolov3_1000.weights'

    #net, classes = load_network(classesFile_face,cfg_face,weights_face)

    #classesFile_hand = 'aug/hand/classes.names'
    #cfg_hand = 'aug/hand/darknet-yolov3.cfg'
    #weights_hand = 'aug/hand/darknet-yolov3_1100.weights'

    #net, classes = load_network(classesFile_hand,cfg_hand,weights_hand)

    #orig_img = cv2.imread("test/test.jpg")
    

    #return execute_get_heatmap(net,classes,blob,orig_img)
