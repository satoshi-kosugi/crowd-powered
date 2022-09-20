import numpy as np
import math
import cv2

def RGB2HSV(image):
    image_hsv = cv2.cvtColor((image[:,:,::-1] * 255.).astype(np.uint8), cv2.COLOR_BGR2HSV) * 1.
    image_hsv[:,:,0] /= 180.
    image_hsv[:,:,1] /= 255.
    image_hsv[:,:,2] /= 255.
    return image_hsv

def HSV2RGB(image):
    image_hsv = image.copy()
    image_hsv[:,:,0] *= 180.
    image_hsv[:,:,1] *= 255.
    image_hsv[:,:,2] *= 255.
    image_bgr = cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR) / 255.
    return image_bgr[:,:,::-1]

def adjust_exposure(image, image_parameters):
    return np.clip(image * (image_parameters + 0.05) * 20, 0, 1)

def adjust_saturation(image, image_parameters):
    return np.clip(image * (image_parameters + 2) / 2, 0, 1)

def adjust_unsharp(applied_v, image_parameters, applied_blur):
    return np.clip(applied_v + ((applied_v - applied_blur) * np.clip(image_parameters, 0, 1) * 3), 0, 1)

def apply_filters_lpf(image, image_parameters_, blur_image_original):
    image_parameters = np.clip(image_parameters_, -1, 1)
    image_rgb = HSV2RGB(image.copy())
    image_rgb = adjust_exposure(image_rgb.copy(), image_parameters[:,:,0:1])
    image = RGB2HSV(image_rgb)
    applied_v = image[:,:,2].copy()
    applied_s = image[:,:,1].copy()
    applied_s = adjust_saturation(applied_s, image_parameters[:,:,1])
    applied_blur = adjust_exposure(blur_image_original, image_parameters[:,:,0])
    applied_v = adjust_unsharp(applied_v, image_parameters[:,:,2], applied_blur)
    applied_image = image.copy()
    applied_image[:,:,2] = applied_v
    applied_image[:,:,1] = applied_s
    return applied_image

def apply_graduated_filters(image, image_parameters, blur_image_original):
    sge, sgs, sgu, x0, x1, y0, y1, ratio_ = image_parameters.tolist()
    x0 += 0.5
    x1 += 0.5
    y0 += 0.5
    y1 += 0.5
    ratio = (ratio_ * (1 - 0.00000001) + 1) / 2

    if x0 == x1 and y0 == y1:
        x1 += 0.00000001
        y1 += 0.00000001
    elif y0 == y1:
        y1 += 0.00000001

    if y1 != y0:
        x2 = x1 * ratio + x0 * (1 - ratio)
        y2 = y1 * ratio + y0 * (1 - ratio)

        a = - (x1 - x0) / (y1 - y0)

        b0 = y0 - a * x0
        b1 = y1 - a * x1
        b2 = y2 - a * x2

        axis = np.zeros([image.shape[0], image.shape[1], 2])
        axis[:,:,0] = np.array([[x / image.shape[1] for x in range(image.shape[1])] for _ in range(image.shape[0])])
        axis[:,:,1] = np.array([[y / image.shape[0] for _ in range(image.shape[1])] for y in range(image.shape[0])])

        s = np.zeros_like(axis[:,:,0])
        flag = np.zeros_like(axis[:,:,0])
        b_ = axis[:,:,1] - a * axis[:,:,0]

        if y1 > y0:
            flag[b_>=b1] = 3
            flag[(b_<=b1)*(b_>=b2)] = 2
            flag[(b_>=b0)*(b_<=b2)] = 1
            flag[b_<=b0] = 0
        else:
            flag[b_<=b1] = 3
            flag[(b_>=b1)*(b_<=b2)] = 2
            flag[(b_<=b0)*(b_>=b2)] = 1
            flag[b_>=b0] = 0

        s[flag==3] = 1
        s[flag==2] = 0.5 + 0.5 * ((b_-b2) / (b1-b2))[flag==2]
        s[flag==1] = 0.5 * ((b_-b0) / (b2-b0))[flag==1]
        s[flag==0] = 0

    se = s * sge
    ss = s * sgs
    su = s * sgu

    return apply_filters_lpf(image, np.concatenate([se[:,:,None], ss[:,:,None], su[:,:,None]], axis=2),
                                            blur_image_original)



def apply_elliptical_filters(image, image_parameters, blur_image_original):
    see, ses, seu, h, k, theta, a, b = image_parameters.tolist()
    h += 0.5
    k += 0.5

    axis = np.zeros([image.shape[0], image.shape[1], 2])
    axis[:,:,0] = np.array([[x / image.shape[1] for x in range(image.shape[1])] for _ in range(image.shape[0])])
    axis[:,:,1] = np.array([[y / image.shape[0] for _ in range(image.shape[1])] for y in range(image.shape[0])])

    s = 1 - \
        (((axis[:,:,0] - h) * math.cos(theta) + (axis[:,:,1] - k) * math.sin(theta)) ** 2) / (a ** 2 + 0.000000001)\
        - (((axis[:,:,0] - h) * math.sin(theta) - (axis[:,:,1] - k) * math.cos(theta)) ** 2) / (b ** 2 + 0.000000001)
    s = np.clip(s, 0, None)

    se = s * see
    ss = s * ses
    su = s * seu

    return apply_filters_lpf(image, np.concatenate([se[:,:,None], ss[:,:,None], su[:,:,None]], axis=2),
                                            blur_image_original)

def apply_cubic10_filters(image, image_parameters, blur_image_original):
    x = np.array([[x / image.shape[1] for x in range(image.shape[1])] for _ in range(image.shape[0])])
    y = np.array([[y / image.shape[0] for _ in range(image.shape[1])] for y in range(image.shape[0])])

    basis = [(x**3), (x**2)*y, (x**2), x*(y**2), x*y, x, y**3, y**2, y, 1]

    se, ss, su = 0, 0, 0
    for i in range(10):
        se += basis[i] * image_parameters[i]
        ss += basis[i] * image_parameters[i+10]
        su += basis[i] * image_parameters[i+20]

    return apply_filters_lpf(image, np.concatenate([se[:,:,None], ss[:,:,None], su[:,:,None]], axis=2),
                                            blur_image_original)

def apply_cubic20_filters(image, image_parameters, blur_image_original):
    x = np.array([[x / image.shape[1] for x in range(image.shape[1])] for _ in range(image.shape[0])])
    y = np.array([[y / image.shape[0] for _ in range(image.shape[1])] for y in range(image.shape[0])])
    i = image[:,:,2]

    basis = [(x**3), (x**2)*y, (x**2)*i, x**2, x*(y**2),\
                x*y*i, x*y, x*(i**2), x*i, x, \
                y**3, (y**2)*i, y**2, y*(i**2), y*i,\
                y, i**3, i**2, i, 1]

    basis = [i, (x**3), (x**2)*y, (x**2)*i, x**2, x*(y**2),\
                x*y*i, x*y, x*(i**2), x*i, x, \
                y**3, (y**2)*i, y**2, y*(i**2), y*i,\
                y, i**3, i**2, 1]

    se, ss, su = 0, 0, 0
    for i in range(10):
        se += basis[i] * image_parameters[i]
        ss += basis[i] * image_parameters[i+10]
        su += basis[i] * image_parameters[i+20]

    return apply_filters_lpf(image, np.concatenate([se[:,:,None], ss[:,:,None], su[:,:,None]], axis=2),
                                            blur_image_original)

def apply_global_filters(image, image_parameters, blur_image_original):
    image_parameters_ = np.ones((image.shape[0], image.shape[1], len(image_parameters)))
    for i in range(image_parameters_.shape[2]):
        image_parameters_[:,:,i] *= image_parameters[i]
    return apply_filters_lpf(image, image_parameters_, blur_image_original)

def apply_filters(filterType, image_hsl, image_parameters, blur_image_original):
    if filterType == "graduated":
        return apply_graduated_filters(image_hsl, image_parameters, blur_image_original)
    elif filterType == "elliptical":
        return apply_elliptical_filters(image_hsl, image_parameters, blur_image_original)
    elif filterType == "cubic10":
        return apply_cubic10_filters(image_hsl, image_parameters, blur_image_original)
    elif filterType == "cubic20":
        return apply_cubic20_filters(image_hsl, image_parameters, blur_image_original)
    elif filterType == "global":
        return apply_global_filters(image_hsl, image_parameters, blur_image_original)
