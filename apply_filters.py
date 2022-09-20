import numpy as np
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

def adjust_exposure_initLIME(image, image_parameters, illumination):
    return np.clip(image / illumination ** 0.7 * (image_parameters + 0.05) * 20, 0, 1)

def adjust_saturation(image, image_parameters):
    return np.clip(image * (image_parameters + 2) / 2, 0, 1)

def adjust_unsharp(applied_v, image_parameters, applied_blur):
    return np.clip(applied_v + ((applied_v - applied_blur) * np.clip(image_parameters, 0, 1) * 3), 0, 1)

def apply_filters_initLIME(image, image_parameters_, blur_image_original, illumination):
    image_parameters = np.clip(image_parameters_, -1, 1) / 3.
    image_rgb = HSV2RGB(image.copy())
    image_rgb = adjust_exposure_initLIME(image_rgb.copy(), image_parameters[:,:,0:1], illumination)
    image = RGB2HSV(image_rgb)
    applied_v = image[:,:,2].copy()
    applied_s = image[:,:,1].copy()
    applied_s = adjust_saturation(applied_s, image_parameters[:,:,1])
    applied_blur = adjust_exposure_initLIME(blur_image_original, image_parameters[:,:,0], illumination[:,:,0])
    applied_v = adjust_unsharp(applied_v, image_parameters[:,:,2], applied_blur)
    applied_image = image.copy()
    applied_image[:,:,2] = applied_v
    applied_image[:,:,1] = applied_s
    return applied_image

def apply_filters(image, image_parameters_, blur_image_original):
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
