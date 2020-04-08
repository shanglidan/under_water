#coding=utf-8
import glob, os, json
from mmdet.ops.nms import nms_wrapper
import numpy as np
from tqdm import tqdm
import cv2
import torch
import argparse
import os
import json
import shutil
import time
import pdb

def merge_3(test_json_1, test_json_2, test_json_3):
    nms_type = 'soft_nms'
    nms_op = getattr(nms_wrapper, nms_type)
    id_to_name = {}
    name_to_id = {}
    images_id = []
    count_id = 1
    index_nms = {1:1}
    index_nms_ = {1:1}
    for image_1 in test_json_1:
        imgid = image_1['image_id']
        if imgid not in images_id:
            images_id.append(imgid)

    for image_1 in test_json_2:
        imgid = image_1['image_id']
        if imgid not in images_id:
            images_id.append(imgid)

    for image_1 in test_json_3:
        imgid = image_1['image_id']
        if imgid not in images_id:
            images_id.append(imgid)

    result = {image_id: [[] for i in range(1)] for image_id in images_id}

    for ann in test_json_1:
        img_id = ann['image_id']
        cls = 0
        score = ann['score']
        bbox = ann['bbox']
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
        bbox = bbox + [score]
    
        result[img_id][cls].append(np.array(bbox)) 
           
    for ann in test_json_2:
        img_id = ann['image_id']
        cls = 0
        score = ann['score']
        bbox = ann['bbox']
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
        bbox = bbox + [score]
        result[img_id][cls].append(np.array(bbox))  

    for ann in test_json_3:
        img_id = ann['image_id']
        cls = 0
        score = ann['score']
        bbox = ann['bbox']
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
        bbox = bbox + [score]
        result[img_id][cls].append(np.array(bbox)) 

    ann = []
    print(len(images_id))
    for image_id in tqdm(images_id):
        det = np.array(result[image_id][0], dtype='float32')    
        if det.shape[0] == 0:
            continue
        cls_dets, _ = nms_op(det, iou_thr=0.5,min_score=0.0001)

        for bbox in cls_dets:
            res_line = {'image_id': image_id, 'category_id': 1, 
                        'bbox': [round(float(bbox[0]), 2), round(float(bbox[1]), 2),  round(float(bbox[2]-bbox[0]),2), round(float(bbox[3]-bbox[1]),2)],
                        'score': float(bbox[4])}
            
            ann.append(res_line)
    return ann

def json2csv(test_json_raw_1, test_json_raw_2, test_json_1, test_json_2, submit_file_name):
    submit_path = 'data/submit/'
    csv_file = open(submit_path + submit_file_name, 'w')
    csv_file.write("name,image_id,confidence,xmin,ymin,xmax,ymax\n")
    imgid2anno = {}
    imgid2name = {}
    # pdb.set_trace()
    for imageinfo in test_json_raw_1['images']:
        imgid = imageinfo['id']
        imgid2name[imgid] = imageinfo['file_name']
    for anno in test_json_1:
        img_id = anno['image_id']
        if img_id not in imgid2anno:
            imgid2anno[img_id] = []
        imgid2anno[img_id].append(anno)
    # pdb.set_trace()
    for imgid, annos in imgid2anno.items():
        for anno in annos:
            xmin, ymin, w, h = anno['bbox']
            xmax = xmin + w
            ymax = ymin + h
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            confidence = anno['score']
            class_id = 1
           # class_name = underwater_classes[class_id-1]
            class_name = 'target'
            print(imgid)
            image_name = imgid2name[imgid]
            image_id = image_name.split('.')[0] + '.xml'
            csv_file.write(class_name + ',' + image_id + ',' + str(confidence) + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + '\n')
    
    imgid2anno = {}
    imgid2name = {}
    for imageinfo in test_json_raw_2['images']:
        imgid = imageinfo['id']
        imgid2name[imgid] = imageinfo['file_name']
    for anno in test_json_2:
        img_id = anno['image_id']
        if img_id not in imgid2anno:
            imgid2anno[img_id] = []
        imgid2anno[img_id].append(anno)
    for imgid, annos in imgid2anno.items():
        for anno in annos:
            xmin, ymin, w, h = anno['bbox']
            xmax = xmin + w
            ymax = ymin + h
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            confidence = anno['score']
            class_id = 1
           # class_name = underwater_classes[class_id-1]
            class_name = 'target'
            image_name = imgid2name[imgid]
            image_id = image_name.split('.')[0] + '.xml'
            csv_file.write(class_name + ',' + image_id + ',' + str(confidence) + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + '\n')    

    csv_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="merge nms")

    parser.add_argument('--test_json_front_1', help='test result json', type=str)
    parser.add_argument('--test_json_front_2', help='test result json', type=str)
    parser.add_argument('--test_json_side_1', help='test result json', type=str)
    parser.add_argument('--test_json_side_2', help='test result json', type=str)
    parser.add_argument('--test_json_side_3', help='test result json', type=str)
    parser.add_argument('--test_json_raw_front', help='test result json', type=str)
    parser.add_argument('--test_json_raw_side', help='test result json', type=str)
    parser.add_argument('--submit_file', help='submit_file_name', type=str)

    args = parser.parse_args()
    test_json_raw_front = json.load(open(args.test_json_raw_front, "r"))
    test_json_raw_side = json.load(open(args.test_json_raw_side, "r"))
    submit_file_name = args.submit_file   
    test_json_front_1 = json.load(open("data/results/" + args.test_json_front_1, "r"))
    test_json_front_2 = json.load(open("data/results/" + args.test_json_front_2, "r"))
    test_json_side_1 = json.load(open("data/results/" + args.test_json_side_1, "r"))
    test_json_side_2 = json.load(open("data/results/" + args.test_json_side_2, "r"))  
    test_json_side_3 = json.load(open("data/results/" + args.test_json_side_3, "r")) 
    side = merge_3(test_json_side_1,test_json_side_2, test_json_side_3)
    front = test_json_front_1
    json2csv(test_json_raw_front, test_json_raw_side, front, side, submit_file_name)

