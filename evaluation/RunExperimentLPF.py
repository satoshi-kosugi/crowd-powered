#! /usr/bin/python

import scipy.io
import numpy
import time
import socket
import datetime
import pickle
import cv2
import argparse
import shutil
from multiprocessing import Pool
import sys
import os
from tqdm import tqdm
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),os.pardir))

###

import datasetAcquisition
import pySequentialLineSearch
import generate_initial_slider
from apply_filters_LPF import apply_filters

###

parser = argparse.ArgumentParser()
parser.add_argument('--label_hitid_record')
parser.add_argument('--image_names')
parser.add_argument('--mode')
parser.add_argument('--filter_type')
args = parser.parse_args()

mode = args.mode
filterType = args.filter_type


print('loading data ...')

if args.image_names is None:
    image_names = []
    for i in range(95):
        image_names.append(str(i))
else:
    image_names = args.image_names.split(",")

def prepare_image(image_name):
    input_image_original, image_name, blur_image_original = datasetAcquisition.readDataLPF(image_name)
    data_tmp = {"input_image_original":input_image_original, \
                "image_name":image_name, "blur_image_original":blur_image_original}

    if filterType == "graduated" or filterType == "elliptical":
        data_tmp["slsoptimizer"] = pySequentialLineSearch.SequentialLineSearchOptimizer(num_dims=8,
            slider_end_selection_strategy=pySequentialLineSearch.SliderEndSelectionStrategy.LastSelection,
            initial_slider_generator=generate_initial_slider.InitialSliderGenerator(8).generate_initial_slider)
    elif filterType == "cubic10":
        data_tmp["slsoptimizer"] = pySequentialLineSearch.SequentialLineSearchOptimizer(num_dims=30,
            slider_end_selection_strategy=pySequentialLineSearch.SliderEndSelectionStrategy.LastSelection,
            initial_slider_generator=generate_initial_slider.InitialSliderGenerator(30).generate_initial_slider)
    elif filterType == "cubic20":
        data_tmp["slsoptimizer"] = pySequentialLineSearch.SequentialLineSearchOptimizer(num_dims=60,
            slider_end_selection_strategy=pySequentialLineSearch.SliderEndSelectionStrategy.LastSelection,
            initial_slider_generator=generate_initial_slider.InitialSliderGenerator(60).generate_initial_slider)
    elif filterType == "global":
        data_tmp["slsoptimizer"] = pySequentialLineSearch.SequentialLineSearchOptimizer(num_dims=3,
            slider_end_selection_strategy=pySequentialLineSearch.SliderEndSelectionStrategy.LastSelection,
            initial_slider_generator=generate_initial_slider.InitialSliderGenerator(3).generate_initial_slider)
    data_tmp["best_point"] = 0.5

    return data_tmp

data = []
for image_name in image_names:
    data.append(prepare_image(image_name))


if mode == "BIQME":
    numIteration = 30
    initLIME = False
    denoise = False
    BIQME_scores = numpy.zeros((numIteration, len(image_names)))
    interp_mode = "cubic"
    if not os.path.exists('BIQME' + str(filterType)):
        os.mkdir('BIQME' + str(filterType))
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.cd('BIQME' + str(filterType))
    eng.addpath('../BIQME/')
    shutil.copyfile("BIQME/model", 'BIQME' + filterType + "/model")
    shutil.copyfile("BIQME/range", 'BIQME' + filterType + "/range")
else:
    raise NotImplementedError()

if args.label_hitid_record is not None:
    with open(args.label_hitid_record, 'rb') as f:
        label_hitid_record = pickle.load(f)
else:
    label_hitid_record = {"label":{}, "HITId":{}, "slider_ends":{}, "approve_reject":{}}
label_hitid_record_name = "log/"+datetime.datetime.strftime(datetime.datetime.now(), '%Y.%m.%d.%H.%M.%S')
if not os.path.exists("log"):
    os.mkdir("log")



for iteration in range(numIteration):
    print("iteration:", iteration)
    scp_image_names = []

    if str(iteration) in label_hitid_record["slider_ends"].keys():
        for data_idx in range(len(data)):
            data[data_idx]["slider_ends"] = label_hitid_record["slider_ends"][str(iteration)][data_idx]
            if data[data_idx]["slider_ends"][0].mean() * 0 != 0:
                data[data_idx]["slider_ends"] = label_hitid_record["slider_ends"][str(iteration-1)][data_idx]
                label_hitid_record["slider_ends"][str(iteration)][data_idx] = data[data_idx]["slider_ends"]
    else:
        label_hitid_record["slider_ends"][str(iteration)] = []
        for data_idx in range(len(data)):
            data[data_idx]["slider_ends"] = data[data_idx]["slsoptimizer"].get_slider_ends()
            if data[data_idx]["slider_ends"][0].mean() * 0 != 0:
                data[data_idx]["slider_ends"] = label_hitid_record["slider_ends"][str(iteration-1)][data_idx]
            label_hitid_record["slider_ends"][str(iteration)].append(data[data_idx]["slider_ends"])
        with open(label_hitid_record_name+".pickle", 'wb') as f:
            pickle.dump(label_hitid_record, f)

    for data_idx in tqdm(range(len(data))):
        t1 = time.time()

        slider_ends = data[data_idx]["slider_ends"]

        min_diff = 1000000000
        min_diff_i = 0

        for i in range(32):
            if interp_mode == "cubic":
                y0 = data[data_idx]["best_point"]
                y1 = numpy.array(slider_ends[0] - y0)
                y2 = numpy.array(y0 - slider_ends[1])
                b = 1 / (numpy.abs(y2/y1) ** (1/3) + 1)
                a = - y1 / (b ** 3)
                a[numpy.abs(y1)<10**-5] = -y2[numpy.abs(y1)<10**-5]
                b[numpy.abs(y1)<10**-5] = 0
                data[data_idx]["filter_params"] = (a * (i / 31. - b) ** 3 + y0) * 2 - 1
                if data[data_idx]["filter_params"].sum() * 0 != 0:
                    import ipdb; ipdb.set_trace()
            else:
                data[data_idx]["filter_params"] = (slider_ends[0] * (1 - i / 31.) + slider_ends[1] * i / 31.) * 2 - 1

            predicted_image = apply_filters(filterType, data[data_idx]["input_image_original"], data[data_idx]["filter_params"], data[data_idx]["blur_image_original"])

            if mode == "BIQME":
                predicted_image_bgr = datasetAcquisition.HSV2BGR(predicted_image).astype(numpy.uint8)
                if not os.path.exists("BIQME"+filterType+"/"):
                    os.mkdir("BIQME"+filterType+"/")
                cv2.imwrite("BIQME"+filterType+"/"+str(i)+".png", predicted_image_bgr)

        if mode == "BIQME":
            if not (str(iteration) in label_hitid_record["label"].keys()):
                while True:
                    try:
                        values = eng.BIQMEmulti(filterType)
                        break
                    except Exception as e:
                        print(e)
                        eng = matlab.engine.start_matlab()
                        eng.cd('BIQME' + filterType)
                        eng.addpath('../')
                values = numpy.array(values)[0]
                min_diff_i = values.argmax()
                min_diff = values.max()
            data[data_idx]["min_diff_i"] = min_diff_i
            BIQME_scores[iteration, data_idx] = min_diff

    if str(iteration) in label_hitid_record["label"].keys():
        results = label_hitid_record["label"][str(iteration)]
    else:
        if mode == "BIQME":
            results = {}
            for data_idx in range(len(data)):
                results[data[data_idx]["image_name"]] = data[data_idx]["min_diff_i"]

        label_hitid_record["label"][str(iteration)] = results
        with open(label_hitid_record_name+".pickle", 'wb') as f:
            pickle.dump(label_hitid_record, f)

    for data_idx in range(len(data)):
        min_diff_i = results[data[data_idx]["image_name"]]
        slider_ends = data[data_idx]["slider_ends"]
        if interp_mode == "cubic":
            y0 = data[data_idx]["best_point"]
            y1 = numpy.array(slider_ends[0] - y0)
            y2 = numpy.array(y0 - slider_ends[1])
            b = 1 / (numpy.abs(y2/y1) ** (1/3) + 1)
            a = - y1 / (b ** 3)
            a[numpy.abs(y1)<10**-5] = -y2[numpy.abs(y1)<10**-5]
            b[numpy.abs(y1)<10**-5] = 0
            data[data_idx]["best_point"] = a * (min_diff_i / 31. - b) ** 3 + y0
        else:
            data[data_idx]["best_point"] = slider_ends[0] * (1 - min_diff_i / 31.) + slider_ends[1] * min_diff_i / 31.

        data[data_idx]["slsoptimizer"].add_line_search_result( \
            data[data_idx]["best_point"], slider_ends[0], slider_ends[1])

        data[data_idx]["filter_params"] = data[data_idx]["best_point"] * 2 - 1
        predicted_image = apply_filters(filterType, data[data_idx]["input_image_original"], data[data_idx]["filter_params"], data[data_idx]["blur_image_original"])

    numpy.save(label_hitid_record_name+"BIQME_scores"+filterType, BIQME_scores)
    print("Average BIQME score:", BIQME_scores[iteration].mean())

if not os.path.exists("results"+mode+filterType+"_LPF/"):
    os.mkdir("results"+mode+filterType+"_LPF/")
save_dir = "results"+mode+filterType+"_LPF/"+os.path.basename(label_hitid_record_name)+"/"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
print("Results are saved in "+save_dir)

for data_idx in range(len(data)):
    predicted_image = apply_filters(filterType, data[data_idx]["input_image_original"], data[data_idx]["filter_params"], data[data_idx]["blur_image_original"])
    predicted_image_bgr = datasetAcquisition.HSV2BGR(predicted_image).astype(numpy.uint8)
    cv2.imwrite(save_dir+data[data_idx]["image_name"]+".png", predicted_image_bgr)

numpy.save(label_hitid_record_name+"BIQME_scores"+filterType, BIQME_scores)
print("BIQME scores are saved as "+label_hitid_record_name+"BIQME_scores"+filterType+".npy")
