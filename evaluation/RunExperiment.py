#! /usr/bin/python

import numpy
import time
import socket
import datetime
import pickle
import cv2
import argparse
import sys
import os
import math
import shutil
from multiprocessing import Pool
import urllib
from tqdm import tqdm
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),os.pardir))

import datasetAcquisition
import pySequentialLineSearch
import generate_initial_slider
from apply_filters import apply_filters, apply_filters_initLIME
import activeLearning.activeLearningGPemocKmeans
from config import activeLearning_config, AMT_config

###

parser = argparse.ArgumentParser()
parser.add_argument('--label_hitid_record')
parser.add_argument('--image_names')
parser.add_argument('--num_pixels', type=int, default=4)
parser.add_argument('--mode')
parser.add_argument('--woactivelearning', action='store_true')
parser.add_argument('--woilluminationmap', action='store_true')
args = parser.parse_args()

sigmaN = activeLearning_config["sigmaN"]
kernel = activeLearning_config["kernel"]
gamma = activeLearning_config["gamma"]
numKernelCores = activeLearning_config["numKernelCores"]
numPixels = args.num_pixels
mode = args.mode


print('loading data ...')

if args.image_names is None:
    image_names = []
    for i in range(95):
        image_names.append(str(i))
else:
    image_names = args.image_names.split(",")

if not os.path.exists("LIME/illumination"):
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.cd('LIME')
    eng.enhanceAll()

def prepare_image(image_name):
    x_original, x_centers, x_counts, input_image_original, x_labeles, image_name, blur_image_original = \
                                            datasetAcquisition.readData(image_name, args.woilluminationmap, args.woactivelearning)
    data_tmp = {"x_original":x_original, "x_centers":x_centers, "x_counts":x_counts,
            "input_image_original":input_image_original, "x_labeles":x_labeles,
            "image_name":image_name, "blur_image_original":blur_image_original}

    if mode == "self":
        data_tmp["preview_images"] = [0 for i in range(32)]

    data_tmp["slsoptimizers"] = []
    data_tmp["initial_slider_generator"] = generate_initial_slider.InitialSliderGenerator(3)

    data_tmp["illumination"] = (cv2.imread("LIME/illumination/"+image_name+".png")[:,:,0:1] / 255.)

    data_tmp["filter_params"] = numpy.zeros_like(input_image_original) * 1.
    data_tmp["regressors"] = numpy.zeros_like(input_image_original) * 1.
    data_tmp["labeleds"] = [numpy.zeros_like(input_image_original.astype(numpy.uint8)[:,:,0]),
                    numpy.zeros_like(input_image_original.astype(numpy.uint8)[:,:,0]),
                    numpy.zeros_like(input_image_original.astype(numpy.uint8)[:,:,0])]
    data_tmp["xTrains"] = []
    data_tmp["xTrainCountss"] = []
    data_tmp["yTrains"] = []
    data_tmp["trainIdxss"] = []
    return data_tmp

cv2.setNumThreads(0)
p = Pool(20)
data = p.map(prepare_image, image_names)
p.close()


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
    initLIME = True
    denoise = True
    numSteps = 4
    interp_mode = "linear"
    from scp_images import scp_images
    scp_images(["validation_images/val_"+str(i)+".jpg" for i in range(5)])
    from throw_task_AMT import *

elif mode == "BIQME":
    numSteps = math.ceil(30 / numPixels)
    initLIME = False
    denoise = False
    BIQME_scores = numpy.zeros((numSteps, numPixels, len(image_names)))
    interp_mode = "cubic"
    if not os.path.exists('BIQME' + str(numPixels)):
        os.mkdir('BIQME' + str(numPixels))
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.cd('BIQME' + str(numPixels))
    eng.addpath('../BIQME/')
    shutil.copyfile("BIQME/model", 'BIQME' + str(numPixels) + "/model")
    shutil.copyfile("BIQME/range", 'BIQME' + str(numPixels) + "/range")

elif mode == "self":
    interp_mode = "cubic"
    initLIME = False
    denoise = False
    numSteps = 7
    cv2.namedWindow('photo', cv2.WINDOW_NORMAL)
    def nothing(x):
        pass
    cv2.createTrackbar("Param (please press \"n\" if finished)","photo",0,31,nothing)


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

if denoise:
    from NBNet.model import UNetD
    import megengine
    from skimage import img_as_float32 as img_as_float
    NBNet = UNetD(3)
    if not os.path.exists("NBNet/NBNet_mge.pkl"):
        url = 'https://github.com/megvii-research/NBNet/releases/download/210418/NBNet_mge.pkl'
        with urllib.request.urlopen(url) as u:
          with open("NBNet/NBNet_mge.pkl", 'bw') as o:
            o.write(u.read())
    with open("NBNet/NBNet_mge.pkl", "rb") as f:
        state = pickle.load(f)
    NBNet.load_state_dict(state["state_dict"])
    NBNet.eval()
    def denoising(noisy_image_, illumination):
        noisy_image = numpy.zeros((512,512,3), dtype=numpy.uint8)
        noisy_image[:noisy_image_.shape[0], :noisy_image_.shape[1]] = noisy_image_
        noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
        noisy_image = img_as_float(noisy_image)
        noisy_image = megengine.tensor(noisy_image.transpose(2,0,1)[None])
        res = NBNet(noisy_image)
        pred = (noisy_image - res)[:, :, :noisy_image_.shape[0], :noisy_image_.shape[1]]
        pred = numpy.clip(numpy.asarray(pred * 255.)[0], 0, 255).transpose(1,2,0).astype(numpy.uint8)
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        pred = pred * (1 - illumination) + noisy_image_ * illumination
        return pred.astype(numpy.uint8)

if mode == "AMT":
    predicted_images = {}
    for data_idx in range(len(data)):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        input_image_original_ = numpy.clip(datasetAcquisition.HSV2BGR(data[data_idx]["input_image_original"]) / data[data_idx]["illumination"] ** 0.7, 0, 255).astype(numpy.uint8)
        if denoise:
            input_image_original_ = denoising(input_image_original_, data[data_idx]["illumination"]).astype(numpy.uint8)
        cv2.imwrite("html_images/"+data[data_idx]["image_name"]+"_"+str(1)+"_"+str(0)+"_"+str(-1)+".jpg", input_image_original_, encode_param)
        predicted_images["html_images/"+data[data_idx]["image_name"]+"_"+str(1)+"_"+str(0)+"_"+str(-1)+".jpg"] = datasetAcquisition.HSV2BGR(data[data_idx]["input_image_original"])

for step in range(numSteps+1):
    for data_idx in range(len(data)):
        data[data_idx]["trainIdxs"] = numpy.array([], dtype="int64")
        data[data_idx]["testIdxs"] = numpy.asarray(range(data[data_idx]["x_centers"].shape[0]))
        data[data_idx]["poolIdxs"] = numpy.delete(numpy.asarray(range(data[data_idx]["x_centers"].shape[0])), data[data_idx]["trainIdxs"])

        data[data_idx]["xTest"] = data[data_idx]["x_centers"][data[data_idx]["testIdxs"],:]
        data[data_idx]["xTrain"] = data[data_idx]["x_centers"][data[data_idx]["trainIdxs"],:]
        data[data_idx]["xTrainCounts"] = data[data_idx]["x_counts"][data[data_idx]["trainIdxs"],:]
        data[data_idx]["yTrain"] = data[data_idx]["x_centers"][data[data_idx]["trainIdxs"],0] #dammy
        data[data_idx]["xPool"] = data[data_idx]["x_centers"][data[data_idx]["poolIdxs"],:]
        try:
            data[data_idx]["xPoolCounts"] = data[data_idx]["x_counts"][data[data_idx]["poolIdxs"],:]
        except:
            import ipdb; ipdb.set_trace()
        data[data_idx]["orgIdxs"] = numpy.asmatrix(range(1, data[data_idx]["xPool"].shape[0] + 1))
        data[data_idx]["regressor"] = activeLearning.activeLearningGPemocKmeans.Regressor(verbose=True)


    for pixel in range(numPixels):
        if step !=0:
            print("step:", step-1, "  pixel:", pixel)
        scp_image_names = []
        for data_idx in range(len(data)):
            scp_image_names.append("html_images/"+data[data_idx]["image_name"]+"_"+str(step)+"_"+str(pixel)+"_"+str(-1)+".jpg")
        interps = {}
        prev_i = {}

        if step != 0:
            if str(step)+"_"+str(pixel) in label_hitid_record["slider_ends"].keys():
                for data_idx in range(len(data)):
                    data[data_idx]["slider_ends"] = label_hitid_record["slider_ends"][str(step)+"_"+str(pixel)][data[data_idx]["image_name"]]
                    if data[data_idx]["slider_ends"][0].mean() * 0 != 0:
                        if pixel == 0:
                            prev_step_pixel = str(step-1)+"_"+str(numPixels-1)
                        else:
                            prev_step_pixel = str(step)+"_"+str(pixel-1)
                        data[data_idx]["slider_ends"] = label_hitid_record["slider_ends"][prev_step_pixel][data[data_idx]["image_name"]]
                        label_hitid_record["slider_ends"][str(step)+"_"+str(pixel)][data[data_idx]["image_name"]] = data[data_idx]["slider_ends"]
            else:
                label_hitid_record["slider_ends"][str(step)+"_"+str(pixel)] = {}
                for data_idx in range(len(data)):
                    data[data_idx]["slider_ends"] = data[data_idx]["slsoptimizers"][pixel].get_slider_ends()

                    if data[data_idx]["slider_ends"][0].mean() * 0 != 0:
                        if pixel == 0:
                            prev_step_pixel = str(step-1)+"_"+str(numPixels-1)
                        else:
                            prev_step_pixel = str(step)+"_"+str(pixel-1)
                        data[data_idx]["slider_ends"] = label_hitid_record["slider_ends"][prev_step_pixel][data[data_idx]["image_name"]]
                    label_hitid_record["slider_ends"][str(step)+"_"+str(pixel)][data[data_idx]["image_name"]] = data[data_idx]["slider_ends"]
                with open(label_hitid_record_name+".pickle", 'wb') as f:
                    pickle.dump(label_hitid_record, f)

        if mode == "BIQME" and step != 0:
            range_ = tqdm(range(len(data)))
        else:
            range_ = range(len(data))
        for data_idx in range_:
            if step == 0:
                if not args.woactivelearning:
                    alScores = data[data_idx]["regressor"].calcAlScores(data[data_idx]["xPool"], data[data_idx]["xPoolCounts"])
                else:
                    alScores = numpy.asmatrix(numpy.zeros((data[data_idx]["xPool"].shape[0], 1)))

                if alScores.shape[0] != data[data_idx]["xPool"].shape[0]:
                    raise Exception('alScores.shape[0] != xPool.shape[0]')

                if alScores.shape[1] != 1:
                    raise Exception('alScores.shape[1] != 1')

                if not numpy.all(numpy.isfinite(alScores)):
                    raise Exception('not numpy.all(numpy.isfinite(alScores))')

                chosenIdx = numpy.argmax(alScores, axis=0).item(0)
                data[data_idx]["chosenIdx"] = chosenIdx

                newX = data[data_idx]["xPool"][chosenIdx,:]
                newXCounts = data[data_idx]["xPoolCounts"][chosenIdx,:]
                data[data_idx]["xTrain"] = numpy.append(data[data_idx]["xTrain"], newX, axis=0)
                data[data_idx]["xTrainCounts"] = numpy.append(data[data_idx]["xTrainCounts"], newXCounts, axis=0)

                data[data_idx]["slsoptimizers"].append(pySequentialLineSearch.SequentialLineSearchOptimizer(num_dims=3,
                                                    slider_end_selection_strategy=pySequentialLineSearch.SliderEndSelectionStrategy.LastSelection,
                                                    initial_slider_generator=data[data_idx]["initial_slider_generator"].generate_initial_slider))
            else:
                data[data_idx]["xTrain"] = data[data_idx]["xTrains"][0]
                data[data_idx]["xTrainCounts"] = data[data_idx]["xTrainCountss"][0]

                slider_ends = data[data_idx]["slider_ends"]

                independent_axis = False
                if independent_axis:
                    for filterIdx in range(3):
                        slider_ends[0][filterIdx] = (data[data_idx]["yTrains"][filterIdx][pixel] + 1) / 2
                        slider_ends[1][filterIdx] = (data[data_idx]["yTrains"][filterIdx][pixel] + 1) / 2
                    slider_ends[0][(step-1)%3] = 0
                    slider_ends[1][(step-1)%3] = 1
                min_diff = 1000000000
                min_diff_i = 0

                interp = []

                for i in range(32):
                    for filterIdx in range(3):
                        yTrainTmp = data[data_idx]["yTrains"][filterIdx].copy()

                        if interp_mode == "cubic":
                            if step == 1:
                                y0 = numpy.array([0.5])
                            else:
                                y0 = numpy.zeros_like(slider_ends[0])
                                for filterIdx_ in range(3):
                                    y0[filterIdx_] = data[data_idx]["yTrains"][filterIdx_][pixel] / 2 + 0.5
                            y1 = numpy.array(slider_ends[0] - y0)
                            y2 = numpy.array(y0 - slider_ends[1])
                            b = 1 / (numpy.abs(y2/y1) ** (1/3) + 1)
                            a = - y1 / (b ** 3)
                            a[numpy.abs(y1)<10**-5] = -y2[numpy.abs(y1)<10**-5]
                            b[numpy.abs(y1)<10**-5] = 0
                            yTrainTmp[pixel] = (a * (i / 31. - b) ** 3 + y0)[filterIdx] * 2 - 1
                            if filterIdx == 0:
                                interp.append(a * (i / 31. - b) ** 3 + y0)
                        elif interp_mode == "linear":
                            yTrainTmp[pixel] = ((slider_ends[0] * (1 - i / 31.) + slider_ends[1] * i / 31.) * 2 - 1)[filterIdx]
                            if filterIdx == 0:
                                interp.append(i / 31.)

                            if i == 0 and filterIdx == 0:
                                if step == 1:
                                    y0 = numpy.array([0.5, 0.5, 0.5])
                                else:
                                    y0 = numpy.zeros_like(slider_ends[0])
                                    for filterIdx_ in range(3):
                                        y0[filterIdx_] = data[data_idx]["yTrains"][filterIdx_][pixel] / 2 + 0.5
                                prev_i[data[data_idx]["image_name"]] = (y0 - slider_ends[0]).sum() / (slider_ends[1] - slider_ends[0]).sum()

                        data[data_idx]["regressor"].train(data[data_idx]["xTrain"], data[data_idx]["xTrainCounts"], yTrainTmp, sigmaN=sigmaN, gamma=gamma, kernel=kernel, numKernelCores=numKernelCores)
                        preds_all = data[data_idx]["regressor"].predict(data[data_idx]["x_original"])
                        data[data_idx]["filter_params"][:,:,filterIdx] = numpy.asarray(preds_all).reshape(data[data_idx]["input_image_original"].shape[:2])

                    if not initLIME:
                        predicted_image = apply_filters(data[data_idx]["input_image_original"], data[data_idx]["filter_params"], data[data_idx]["blur_image_original"])
                    else:
                        predicted_image = apply_filters_initLIME(data[data_idx]["input_image_original"], data[data_idx]["filter_params"], data[data_idx]["blur_image_original"], data[data_idx]["illumination"])
                    if mode == "AMT":
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                        if denoise:
                            cv2.imwrite("html_images/"+data[data_idx]["image_name"]+"_"+str(step)+"_"+str(pixel)+"_"+str(i)+".jpg", denoising(datasetAcquisition.HSV2BGR(predicted_image), data[data_idx]["illumination"]), encode_param)
                        else:
                            cv2.imwrite("html_images/"+data[data_idx]["image_name"]+"_"+str(step)+"_"+str(pixel)+"_"+str(i)+".jpg", datasetAcquisition.HSV2BGR(predicted_image), encode_param)
                        scp_image_names.append("html_images/"+data[data_idx]["image_name"]+"_"+str(step)+"_"+str(pixel)+"_"+str(i)+".jpg")
                        predicted_images["html_images/"+data[data_idx]["image_name"]+"_"+str(step)+"_"+str(pixel)+"_"+str(i)+".jpg"] = datasetAcquisition.HSV2BGR(predicted_image)

                    elif mode == "BIQME":
                        predicted_image_bgr = datasetAcquisition.HSV2BGR(predicted_image).astype(numpy.uint8)
                        if denoise:
                            predicted_image_bgr = denoising(predicted_image_bgr, data[data_idx]["illumination"])
                        if not os.path.exists("BIQME"+str(numPixels)+"/"):
                            os.mkdir("BIQME"+str(numPixels)+"/")
                        cv2.imwrite("BIQME"+str(numPixels)+"/"+str(i)+".png", predicted_image_bgr)

                    elif mode == "self":
                        predicted_image_bgr = datasetAcquisition.HSV2BGR(predicted_image).astype(numpy.uint8)
                        if denoise:
                            predicted_image_bgr = denoising(predicted_image_bgr, data[data_idx]["illumination"])
                        data[data_idx]["preview_images"][i] = predicted_image_bgr

                if interp_mode == "cubic":
                    for filterIdx in range(3):
                        interp_ = numpy.array(interp)[:, filterIdx]
                        if interp_[0] != interp_[-1]:
                            interp_ = (interp_ - interp_[0]) / (interp_[-1] - interp_[0])
                            interps[data[data_idx]["image_name"]] = interp_
                            break
                        elif filterIdx == 2:
                            interps[data[data_idx]["image_name"]] = interp_ * 0

                elif interp_mode == "linear" or interp_mode == "linear_adjust" or interp_mode == "linear_adjust2":
                    interp = numpy.array(interp)
                    interp = (interp - interp[0]) / (interp[-1] - interp[0])
                    interps[data[data_idx]["image_name"]] = interp

                if mode == "BIQME":
                    if not (str(step)+"_"+str(pixel) in label_hitid_record["label"].keys()):
                        while True:
                            try:
                                values = eng.BIQMEmulti(numPixels)
                                break
                            except Exception as e:
                                print(e)
                                eng = matlab.engine.start_matlab()
                                eng.cd('BIQME' + str(numPixels))
                                eng.addpath('../')
                        values = numpy.array(values)[0]
                        min_diff_i = values.argmax()
                        min_diff = values.max()
                    data[data_idx]["min_diff_i"] = min_diff_i
                    BIQME_scores[step-1, pixel, data_idx] = min_diff

        if mode == "BIQME":
            if step != 0:
                print("Average BIQME score:", BIQME_scores[step-1, pixel].mean())
            numpy.save(label_hitid_record_name+"BIQME_scores"+str(numPixels), BIQME_scores)

        if step != 0:
            if str(step)+"_"+str(pixel) in label_hitid_record["label"].keys():
                results = label_hitid_record["label"][str(step)+"_"+str(pixel)]
                if mode == "AMT":
                    scp_images(scp_image_names)
            else:
                if mode == "AMT":
                    scp_images(scp_image_names)
                    default_values = []
                    if step == 0+1:
                        for idx in range(len(image_names)):
                            default_values.append(0)
                    else:
                        for image_name in image_names:
                            default_values.append(0)
                    results = throw_task_AMT(scp_image_names, image_names, label_hitid_record, label_hitid_record_name, AMT_config, default_values, interps, prev_i, predicted_images)
                elif mode == "BIQME":
                    results = {}
                    for data_idx in range(len(data)):
                        results[data[data_idx]["image_name"]] = data[data_idx]["min_diff_i"]
                elif mode == "self":
                    results = {}
                    for data_idx in range(len(data)):
                        while(1):
                            min_diff_i = cv2.getTrackbarPos("Param (please press \"n\" if finished)", "photo")
                            cv2.imshow('photo', data[data_idx]["preview_images"][min_diff_i] / 255.)
                            if (cv2.waitKey(1) & 0xFF) == 110:
                                break
                        results[data[data_idx]["image_name"]] = min_diff_i

                label_hitid_record["label"][str(step)+"_"+str(pixel)] = results
                with open(label_hitid_record_name+".pickle", 'wb') as f:
                    pickle.dump(label_hitid_record, f)

        for data_idx in range(len(data)):
            if step == 0:
                data[data_idx]["yTrain"] = numpy.append(data[data_idx]["yTrain"], numpy.asarray([[0]]), axis=0)
                data[data_idx]["regressor"].train(data[data_idx]["xTrain"], data[data_idx]["xTrainCounts"], data[data_idx]["yTrain"], sigmaN=sigmaN, gamma=gamma, kernel=kernel, numKernelCores=numKernelCores)
            else:
                min_diff_i = results[data[data_idx]["image_name"]]
                slider_ends = data[data_idx]["slider_ends"]

                if interp_mode == "cubic":
                    if step == 1:
                        y0 = numpy.array([0.5])
                    else:
                        y0 = numpy.zeros_like(slider_ends[0])
                        for filterIdx_ in range(3):
                            y0[filterIdx_] = data[data_idx]["yTrains"][filterIdx_][pixel] / 2 + 0.5
                    y1 = numpy.array(slider_ends[0] - y0)
                    y2 = numpy.array(y0 - slider_ends[1])
                    b = 1 / (numpy.abs(y2/y1) ** (1/3) + 1)
                    a = - y1 / (b ** 3)
                    a[numpy.abs(y1)<10**-5] = -y2[numpy.abs(y1)<10**-5]
                    b[numpy.abs(y1)<10**-5] = 0

                    for filterIdx in range(3):
                        if mode == "AMT":
                            data[data_idx]["yTrains"][filterIdx][pixel] = \
                                ((slider_ends[0] * (1 - min_diff_i / 31.) + slider_ends[1] * min_diff_i / 31.) * 2 - 1)[filterIdx]
                        else:
                            data[data_idx]["yTrains"][filterIdx][pixel] = \
                                (a * (min_diff_i / 31. - b) ** 3 + y0)[filterIdx] * 2 - 1
                    data[data_idx]["slsoptimizers"][pixel].add_line_search_result( \
                        (numpy.array(data[data_idx]["yTrains"])[:,pixel,0] + 1) / 2, \
                        slider_ends[0], slider_ends[1])
                elif interp_mode == "linear" or interp_mode == "linear_adjust" or interp_mode == "linear_adjust2":
                    for filterIdx in range(3):
                        data[data_idx]["yTrains"][filterIdx][pixel] = \
                            ((slider_ends[0] * (1 - min_diff_i / 31.) + slider_ends[1] * min_diff_i / 31.) * 2 - 1)[filterIdx]
                    data[data_idx]["slsoptimizers"][pixel].add_line_search_result( \
                        (numpy.array(data[data_idx]["yTrains"])[:,pixel,0] + 1) / 2, \
                        slider_ends[0], slider_ends[1])

            chosenIdx = data[data_idx]["chosenIdx"]
            data[data_idx]["xPool"] = numpy.delete(data[data_idx]["xPool"], (chosenIdx), axis=0)
            data[data_idx]["xPoolCounts"] = numpy.delete(data[data_idx]["xPoolCounts"], (chosenIdx), axis=0)
            data[data_idx]["orgIdxs"] = numpy.delete(data[data_idx]["orgIdxs"], (chosenIdx), axis=1)
            data[data_idx]["trainIdxs"] = numpy.append(data[data_idx]["trainIdxs"], [data[data_idx]["poolIdxs"][chosenIdx]], axis=0)
            data[data_idx]["poolIdxs"] = numpy.delete(data[data_idx]["poolIdxs"], (chosenIdx), axis=0)

            t2 = time.time()

            if step != 0:
                for filterIdx in range(3):
                    data[data_idx]["regressor"].train(data[data_idx]["xTrain"], data[data_idx]["xTrainCounts"], data[data_idx]["yTrains"][filterIdx], sigmaN=sigmaN, gamma=gamma, kernel=kernel, numKernelCores=numKernelCores)
                    preds_all = data[data_idx]["regressor"].predict(data[data_idx]["x_original"])
                    data[data_idx]["filter_params"][:,:,filterIdx] = numpy.asarray(preds_all).reshape(data[data_idx]["input_image_original"].shape[:2])
            if not initLIME:
                predicted_image = apply_filters(data[data_idx]["input_image_original"], data[data_idx]["filter_params"], data[data_idx]["blur_image_original"])
            else:
                predicted_image = apply_filters_initLIME(data[data_idx]["input_image_original"], data[data_idx]["filter_params"], data[data_idx]["blur_image_original"], data[data_idx]["illumination"])
            labeled = numpy.zeros(data[data_idx]["input_image_original"].shape[0]*data[data_idx]["input_image_original"].shape[1], dtype=numpy.uint8)
            if step == 0:
                for i in range(data[data_idx]["trainIdxs"].shape[0]):
                    labeled[numpy.where(data[data_idx]["x_labeles"] == data[data_idx]["trainIdxs"][i])] = 1
            else:
                for i in range(data[data_idx]["trainIdxs"].shape[0]):
                    labeled[numpy.where(data[data_idx]["x_labeles"] == data[data_idx]["trainIdxss"][0][i])] = 1

            for filterIdx in range(3):
                data[data_idx]["labeleds"][filterIdx] = labeled.reshape(data[data_idx]["input_image_original"].shape[:2])


            if mode == "AMT":
                if step != 0:
                    if not denoise:
                        if pixel == numPixels-1:
                            cv2.imwrite("html_images/"+data[data_idx]["image_name"]+"_"+str(step+1)+"_"+str(0)+"_"+str(-1)+".jpg", datasetAcquisition.HSV2BGR(predicted_image), encode_param)
                            predicted_images["html_images/"+data[data_idx]["image_name"]+"_"+str(step+1)+"_"+str(0)+"_"+str(-1)+".jpg"] = datasetAcquisition.HSV2BGR(predicted_image)
                        else:
                            cv2.imwrite("html_images/"+data[data_idx]["image_name"]+"_"+str(step)+"_"+str(pixel+1)+"_"+str(-1)+".jpg", datasetAcquisition.HSV2BGR(predicted_image), encode_param)
                            predicted_images["html_images/"+data[data_idx]["image_name"]+"_"+str(step)+"_"+str(pixel+1)+"_"+str(-1)+".jpg"] = datasetAcquisition.HSV2BGR(predicted_image)
                    else:
                        if pixel == numPixels-1:
                            cv2.imwrite("html_images/"+data[data_idx]["image_name"]+"_"+str(step+1)+"_"+str(0)+"_"+str(-1)+".jpg", denoising(datasetAcquisition.HSV2BGR(predicted_image), data[data_idx]["illumination"]), encode_param)
                            predicted_images["html_images/"+data[data_idx]["image_name"]+"_"+str(step+1)+"_"+str(0)+"_"+str(-1)+".jpg"] = datasetAcquisition.HSV2BGR(predicted_image)
                        else:
                            cv2.imwrite("html_images/"+data[data_idx]["image_name"]+"_"+str(step)+"_"+str(pixel+1)+"_"+str(-1)+".jpg", denoising(datasetAcquisition.HSV2BGR(predicted_image), data[data_idx]["illumination"]), encode_param)
                            predicted_images["html_images/"+data[data_idx]["image_name"]+"_"+str(step)+"_"+str(pixel+1)+"_"+str(-1)+".jpg"] = datasetAcquisition.HSV2BGR(predicted_image)

        if mode == "AMT":
            pop_image_names = []
            for image_name in predicted_images.keys():
                step_ = image_name.split("/")[1].split("_")[1]
                pixel_ = image_name.split("/")[1].split("_")[2]
                if step_ == step and pixel_ == pixel:
                    pop_image_names.append(image_name)
            for image_names in pop_image_names:
                predicted_images.pop(image_name)

    for data_idx in range(len(data)):
        if step == 0:
            data[data_idx]["xTrains"].append(data[data_idx]["xTrain"])
            data[data_idx]["xTrainCountss"].append(data[data_idx]["xTrainCounts"])
            for filterIdx in range(3):
                data[data_idx]["yTrains"].append(data[data_idx]["yTrain"].copy())
            data[data_idx]["trainIdxss"].append(data[data_idx]["trainIdxs"])


if not os.path.exists("results"+mode+str(numPixels)+"/"):
    os.mkdir("results"+mode+str(numPixels)+"/")
save_dir = "results"+mode+str(numPixels)+"/"+os.path.basename(label_hitid_record_name)+"/"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
print("Results are saved in "+save_dir)

for data_idx in range(len(data)):
    if not initLIME:
        predicted_image = apply_filters(data[data_idx]["input_image_original"], data[data_idx]["filter_params"], data[data_idx]["blur_image_original"])
    else:
        predicted_image = apply_filters_initLIME(data[data_idx]["input_image_original"], data[data_idx]["filter_params"], data[data_idx]["blur_image_original"], data[data_idx]["illumination"])
    predicted_image_bgr = datasetAcquisition.HSV2BGR(predicted_image).astype(numpy.uint8)
    if denoise:
        predicted_image_bgr = denoising(predicted_image_bgr, data[data_idx]["illumination"]).astype(numpy.uint8)

    cv2.imwrite(save_dir+data[data_idx]["image_name"]+".png", predicted_image_bgr)

if mode == "BIQME":
    numpy.save(label_hitid_record_name+"BIQME_scores"+str(numPixels), BIQME_scores)
    print("BIQME scores are saved as "+label_hitid_record_name+"BIQME_scores"+str(numPixels)+".npy")

if mode == "AMT":
    unblock_workers(label_hitid_record["blocked_workers"], AMT_config)
