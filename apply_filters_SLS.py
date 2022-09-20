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
    return np.clip(image * (image_parameters + 0.5) * 2, 0, 1)

def adjust_saturation(image, image_parameters):
    return np.clip(image * (image_parameters + 4) / 4, 0, 1)

def changeColorBalance(image_rgb, color_balance):
    lightness = (image_rgb.max(axis=2, keepdims=True) + image_rgb.min(axis=2, keepdims=True)) / 2.
    value = image_rgb.max(axis=2, keepdims=True)

    a = 0.25
    b = 0.333
    scale = 0.7

    midtones = (np.clip((lightness - b) /  a + 0.5, 0.0, 1.0) * \
        np.clip((lightness + b - 1.0) / -a + 0.5, 0.0, 1.0) * scale) * color_balance[None][None]

    newColor = image_rgb + midtones
    newColor = np.clip(newColor, 0.0, 1.0)

    newHsv = RGB2HSV(newColor)
    newHsv[:,:,2:3] = value
    return HSV2RGB(newHsv)

def apply_filters(image_hsv, image_parameters):
    brightness = image_parameters[0] - 0.5
    contrast = image_parameters[1] - 0.5
    saturation = image_parameters[2] - 0.5
    color_balance = image_parameters[3:6] - 0.5

    image_rgb = HSV2RGB(image_hsv)

    image_rgb = changeColorBalance(image_rgb, color_balance)

    image_rgb = image_rgb * (1. + brightness)

    cont = np.tan((contrast + 1.0) * 3.1415926535 * 0.25)
    image_rgb = (image_rgb - 0.5) * cont + 0.5

    image_rgb = np.clip(image_rgb, 0.0, 1.0)

    image_hsv = RGB2HSV(image_rgb)
    image_hsv[:,:,1] *= (saturation + 1.0)
    image_hsv = np.clip(image_hsv, 0.0, 1.0)

    return image_hsv
