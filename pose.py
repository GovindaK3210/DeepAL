import sys
sys.path.insert(1,'./AlphaPose')

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from AlphaPose.opt import opt

from AlphaPose.dataloader import DetectionLoader, DetectionProcessor, DataWriter, Mscoco
from AlphaPose.modifieddataloader import ModifiedImageLoader
from AlphaPose.yolo.util import write_results, dynamic_write_results
from AlphaPose.SPPE.src.main_fast_inference import *

import cv2
import os

#from AlphaPose.tqdm import tqdm
import time
from AlphaPose.fn import getTime

from PIL import Image,ImageDraw

from AlphaPose.pPose_nms import pose_nms, write_json

args = opt
args.dataset = 'coco'
#if not args.sp:
#    torch.multiprocessing.set_start_method('forkserver', force=True)
#    torch.multiprocessing.set_sharing_strategy('file_system')


if opt.vis_fast:
    from fn import vis_frame_fast as vis_frame
else:
    from fn import vis_frame



def get_heatmap_pose(keypoints, img_w, img_h):
        
        heatmaps = [ Image.new('L', (img_w, img_h)) for x in range(17) ]
        #print ("Inside get_heatmap_pose , heatmap shape"+str (len(heatmaps)))
        #print ("keypoint [0] "+str(keypoints[0]))
        for i in range(0,len(keypoints),3):
           xmin = keypoints[i] - 5
           ymin = keypoints[i+1] - 5
           xmax = keypoints[i] + 5
           ymax = keypoints[i+1] + 5

           hmDraw = ImageDraw.Draw(heatmaps[i//3]) 
           hmDraw.rectangle([xmin,ymin,xmax,ymax], fill ="#ffffff")
        return heatmaps


def execute_pose_heatmaps(det_processor, pose_model):



    #inputpath = args.inputpath
    #inputlist = args.inputlist
    mode = args.mode
    #if not os.path.exists(args.outputpath):
    #    os.mkdir(args.outputpath)

    #if len(inputlist):
    #    im_names = open(inputlist, 'r').readlines()
    #elif len(inputpath) and inputpath != '/':
    #    for root, dirs, files in os.walk(inputpath):
    #        im_names = files
    #else:
    #    raise IOError('Error: must contain either --indir/--list')

    # Read Image
    #image = cv2.imread("test/163.jpg")

    # Load input images
    #data_loader = ModifiedImageLoader(image, batchSize=args.detbatch, format='yolo').start()

    # Load detection loader
    #print('Loading YOLO model..')
    #sys.stdout.flush()
    ##det_loader = DetectionLoader(data_loader, batchSize=args.detbatch).start()
    #det_processor = DetectionProcessor(det_loader).start()
    
    # Load pose model
    #pose_dataset = Mscoco()
    #if args.fast_inference:
    #    pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    #else:
    #    pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    #pose_model.cuda()
    #pose_model.eval()

    #runtime_profile = {
    #    'dt': [],
    #    'pt': [],
    #    'pn': []
    #}

    data_len = 1
    #im_names_desc = tqdm(range(data_len))

    batchSize = 1
    #for i in im_names_desc:
    #    start_time = getTime()
    with torch.no_grad():
        (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
        if boxes is None or boxes.nelement() == 0:
            print("No Boxes")
            return get_heatmap_pose( [], 0, 0)
            
        #ckpt_time, det_time = getTime(start_time)
        #runtime_profile['dt'].append(det_time)
        # Pose Estimation
        
        datalen = inps.size(0)
        leftover = 0
        if (datalen) % batchSize:
            leftover = 1
        num_batches = datalen // batchSize + leftover
        hm = []
        for j in range(num_batches):
            inps_j = inps[j*batchSize:min((j +  1)*batchSize, datalen)].cuda()
            hm_j = pose_model(inps_j)
            hm.append(hm_j)
        hm = torch.cat(hm)
        #ckpt_time, pose_time = getTime(ckpt_time)
        #runtime_profile['pt'].append(pose_time)
        hm = hm.cpu()
        #writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])
        
        
        if opt.matching:
            preds = getMultiPeakPrediction(hm, pt1.numpy(), pt2.numpy(), opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
            result = matching(boxes, scores.numpy(), preds)
        else:
            preds_hm, preds_img, preds_scores = getPrediction(hm, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
            result = pose_nms(boxes, scores, preds_img, preds_scores)

        # To Print Resutl Image
        #result = {
        #    'imgname': im_name,
        #    'result': result
        #}
        #img = vis_frame(orig_img, {'imgname':'test/image.jpg','result':result})
        #cv2.imwrite("test/pred.jpg",img)
    
        #print(result)
        result_final = []
        if (result is not None and len (result)>0):
            resultf= torch.cat((result[0]['keypoints'], result[0]['kp_score']),1).view(51)
            result_final.extend(resultf)
        print(result_final)
        
        return get_heatmap_pose(result_final,orig_img.shape[1],orig_img.shape[0])
        
        