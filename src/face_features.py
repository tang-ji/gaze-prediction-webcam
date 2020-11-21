import numpy as np
import matplotlib.pyplot as plt
import cv2

from src.tools import *
from src.config import *

def extract_image_features(img):
    try:
        preds = landmarks_detector.get_landmarks(img)
        left_eye = get_bound(preds[36:42], 0.1, 0.4)
        right_eye = get_bound(preds[42:48], 0.1, 0.4)
        face = get_bound(preds, 0, 0)
        eye_images = [crop_box(img, left_eye), crop_box(img, right_eye)]
        face_image = crop_box(img, face)
        frameC = 1
        if len(img.shape) == 3 and img.shape[2] == 3:
            frameC = 3
        face_grid = get_face_grid([face[1], face[3], face[0]-face[1], face[2]-face[3]], img.shape[1], img.shape[0], frameC, mask_size)
        return [face_image, eye_images, face_grid]
    except Exception as e:
        print(e)
        return None

def get_face_grid(face, frameW, frameH, frameC, mask_size):
    faceX,faceY,faceW,faceH = face
    return faceGridFromFaceRect(frameW, frameH, frameC, mask_size[0], mask_size[1], faceX, faceY, faceW, faceH, False)

####################
# Given face detection data, generate face grid data.
#
# Input Parameters:
# - frameW/H: The frame in which the detections exist
# - gridW/H: The size of the grid (typically same aspect ratio as the
#     frame, but much smaller)
# - labelFaceX/Y/W/H: The face detection (x and y are 0-based image
#     coordinates)
# - parameterized: Whether to actually output the grid or just the
#     [x y w h] of the 1s square within the gridW x gridH grid.
########
def faceGridFromFaceRect(frameW, frameH, frameC, gridW, gridH, labelFaceX, labelFaceY, labelFaceW, labelFaceH, parameterized):

    scaleX = gridW / frameW
    scaleY = gridH / frameH
    
    if parameterized:
        labelFaceGrid = np.zeros(4)
    else:
        labelFaceGrid = np.zeros(gridW * gridH)
    
    grid = np.zeros((gridH, gridW))

    # Use one-based image coordinates.
    xLo = round(labelFaceX * scaleX)
    yLo = round(labelFaceY * scaleY)
    w = round(labelFaceW * scaleX)
    h = round(labelFaceH * scaleY)

    if parameterized:
        labelFaceGrid = [xLo, yLo, w, h]
    else:
        xHi = xLo + w
        yHi = yLo + h

        # Clamp the values in the range.
        xLo = int(min(gridW, max(0, xLo)))
        xHi = int(min(gridW, max(0, xHi)))
        yLo = int(min(gridH, max(0, yLo)))
        yHi = int(min(gridH, max(0, yHi)))

        faceLocation = np.ones((yHi - yLo, xHi - xLo))
        grid[yLo:yHi, -xHi:-xLo] = faceLocation
    
    if frameC == 3:
        return cv2.cvtColor(np.array(grid*255, dtype=np.uint8),cv2.COLOR_GRAY2RGB)
    else:
        return np.array(grid*255, dtype=np.uint8)
######################

def set_title_and_hide_axis(title):
    plt.title(title)
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)

def show_extraction_results(face_features):
    face_features = np.asarray(face_features)
    plt.figure(figsize=(10,10))
    face_image, eye_images, face_grid = face_features
    plt.figure()
    set_title_and_hide_axis('Extracted face image')
    plt.imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB), interpolation="bicubic")
    plt.figure()
    set_title_and_hide_axis('Face grid')
    plt.imshow(face_grid)

    for eye_image in eye_images:
        plt.figure()

        set_title_and_hide_axis('Extracted eye image')
        plt.imshow(cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB), interpolation="bicubic")
        
def merge_results(window, face_features):
    if face_features is not None:
        if len(face_features[1]) > 0:
            window[:img_size[0], :img_size[1]] = cv2.resize(face_features[1][0], img_size, interpolation = cv2.INTER_AREA)
            window[img_size[0]:, :img_size[1]] = cv2.resize(face_features[1][1], img_size, interpolation = cv2.INTER_AREA)
        if face_features[0] is not None:
            window[:img_size[0], img_size[1]:] = cv2.resize(face_features[0], img_size, interpolation = cv2.INTER_AREA)
        if face_features[2] is not None:
            window[img_size[0]:, img_size[1]:] = cv2.resize(face_features[2], img_size, interpolation = cv2.INTER_AREA)
