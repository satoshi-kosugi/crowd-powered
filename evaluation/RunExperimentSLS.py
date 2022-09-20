#! /usr/bin/python

import scipy.io
import numpy
import time
import socket
import datetime
import pickle
import cv2
import argparse

###

import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),os.pardir))

###

import datasetAcquisition
from scp_images import scp_images
from throw_task_AMT_SLS import *
import pySequentialLineSearch
import generate_initial_slider
from apply_filters_SLS import apply_filters
from config import AMT_config

###

parser = argparse.ArgumentParser()
parser.add_argument('--label_hitid_record')
parser.add_argument('--image_names')
parser.add_argument('--validation_image_name')
parser.add_argument('--mode')
args = parser.parse_args()

mode = args.mode
###

if mode == "AMT":
    if AMT_config["sandbox"]:
        print("This process is performed on sandbox.")
    else:
        print("")
        print("[[Warning]] This process will spend real money. Check the following config and type yes if there are no problems.")
        for key in AMT_config.keys():
            print(key, ":", AMT_config[key])
        if input() != "yes":
            print("bye")
            exit()
    numIteration = 15
else:
    raise NotImplementedError()


if args.image_names is None:
    image_names = []
    for i in range(95):
        image_names.append(str(i))
else:
    image_names = args.image_names.split(",")

def prepare_image(image_name):
    input_image_original, image_name = datasetAcquisition.readDataSLS(image_name)
    data_tmp = {"input_image_original":input_image_original, "image_name":image_name}

    data_tmp["slsoptimizer"] = pySequentialLineSearch.SequentialLineSearchOptimizer(num_dims=6)
    return data_tmp

data = []
for image_name in image_names:
    data.append(prepare_image(image_name))

if args.label_hitid_record is not None:
    with open(args.label_hitid_record, 'rb') as f:
        label_hitid_record = pickle.load(f)
else:
    label_hitid_record = {"label":{}, "HITId":{}, "slider_ends":{}, "approve_reject":{}, "blocked_workers":[]}
label_hitid_record_name = "log/"+datetime.datetime.strftime(datetime.datetime.now(), '%Y.%m.%d.%H.%M.%S')
if not os.path.exists("log"):
    os.mkdir("log")
if not os.path.exists("html_images"):
    os.mkdir("html_images")
if not os.path.exists("html"):
    os.mkdir("html")


for iteration in range(numIteration):
    print("iteration:", iteration)
    scp_image_names = []

    if str(iteration) in label_hitid_record["slider_ends"].keys():
        for data_idx in range(len(data)):
            data[data_idx]["slider_ends"] = label_hitid_record["slider_ends"][str(iteration)][data_idx]
    else:
        label_hitid_record["slider_ends"][str(iteration)] = []
        for data_idx in range(len(data)):
            data[data_idx]["slider_ends"] = data[data_idx]["slsoptimizer"].get_slider_ends()
            label_hitid_record["slider_ends"][str(iteration)].append(data[data_idx]["slider_ends"])
        with open(label_hitid_record_name+".pickle", 'wb') as f:
            pickle.dump(label_hitid_record, f)

    for data_idx in range(len(data)):
        t1 = time.time()

        slider_ends = data[data_idx]["slider_ends"]

        min_diff = 1000000000
        min_diff_i = 0
        for i in range(32):
            data[data_idx]["filter_params"] = (slider_ends[0] * (1 - i / 31.) + slider_ends[1] * i / 31.)
            predicted_image = apply_filters(data[data_idx]["input_image_original"], data[data_idx]["filter_params"])

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            cv2.imwrite("html_images/"+data[data_idx]["image_name"]+"_"+str(iteration)+"_"+str(i)+".jpg", datasetAcquisition.HSV2BGR(predicted_image), encode_param)
            scp_image_names.append("html_images/"+data[data_idx]["image_name"]+"_"+str(iteration)+"_"+str(i)+".jpg")

    if str(iteration) in label_hitid_record["label"].keys():
        results = label_hitid_record["label"][str(iteration)]
    else:
        scp_images(scp_image_names)
        default_values = []
        for image_name in image_names:
            default_values.append(0)

        results = throw_task_AMT(scp_image_names, image_names, label_hitid_record, label_hitid_record_name, AMT_config, default_values)

        label_hitid_record["label"][str(iteration)] = results
        with open(label_hitid_record_name+".pickle", 'wb') as f:
            pickle.dump(label_hitid_record, f)

    for data_idx in range(len(data)):
        min_diff_i = results[data[data_idx]["image_name"]]
        slider_ends = data[data_idx]["slider_ends"]
        data[data_idx]["slsoptimizer"].add_line_search_result( \
            (slider_ends[0] * (1 - min_diff_i / 31.) + slider_ends[1] * min_diff_i / 31.), \
            slider_ends[0], slider_ends[1])

        data[data_idx]["filter_params"] = (slider_ends[0] * (1 - min_diff_i / 31.) + slider_ends[1] * min_diff_i / 31.)
        predicted_image = apply_filters(data[data_idx]["input_image_original"], data[data_idx]["filter_params"])


if not os.path.exists("results"+mode+"_SLS/"):
    os.mkdir("results"+mode+"_SLS/")
save_dir = "results"+mode+"_SLS/"+os.path.basename(label_hitid_record_name)+"/"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
print("Results are saved in "+save_dir)

for data_idx in range(len(data)):
    predicted_image = apply_filters(data[data_idx]["input_image_original"], data[data_idx]["filter_params"])
    predicted_image_bgr = datasetAcquisition.HSV2BGR(predicted_image).astype(numpy.uint8)
    cv2.imwrite(save_dir+data[data_idx]["image_name"]+".png", predicted_image_bgr)

unblock_workers(label_hitid_record["blocked_workers"], AMT_config)
