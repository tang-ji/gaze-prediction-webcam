import numpy as np
import matplotlib.pyplot as plt
import cv2

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

def extract_face_features(face, img, gray):
    [x,y,w,h] = face
    roi_gray = gray[y:y+h, x:x+w]
    face_image = np.copy(img[y:y+h, x:x+w])
    
    eyes = eye_cascade.detectMultiScale(roi_gray)
    eye_images = []
    for (ex,ey,ew,eh) in eyes:
        eye_images.append(np.copy(img[y+ey:y+ey+eh,x+ex:x+ex+ew]))
                
    roi_color = img[y:y+h, x:x+w]
    for (ex,ey,ew,eh) in eyes:
        return face_image, eye_images

def get_face_grid(face, frameW, frameH, gridSize):
    faceX,faceY,faceW,faceH = face
    return faceGridFromFaceRect(frameW, frameH, gridSize, gridSize, faceX, faceY, faceW, faceH, False)

def extract_image_features(full_img_path):
    try:
        img = cv2.imread(full_img_path)
    except:
        img = full_img_path
    gray = img
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        pass
    face_detections = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    try:
        [x,y,w,h] = face_detections[0]
        face = [x,y,w,h]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face_image, eye_images = extract_face_features(face, img, gray)
        face_grid = get_face_grid(face, img.shape[1], img.shape[0], 25)
        face_features = [face_image, eye_images, face_grid]
        return face_features
    except:
        return None

gridSize = 25
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
def faceGridFromFaceRect(frameW, frameH, gridW, gridH, labelFaceX, labelFaceY, labelFaceW, labelFaceH, parameterized):

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
        grid[yLo:yHi, xLo:xHi] = faceLocation

        # Flatten the grid.
        grid = np.transpose(grid)
        labelFaceGrid = grid.flatten()
        
    return labelFaceGrid
######################
def set_title_and_hide_axis(title):
    plt.title(title)
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)

def render_face_grid(face_grid):
    face_grid = np.asarray(face_grid)
    to_print = np.copy(face_grid)
    result_image = np.copy(to_print).reshape(25, 25).transpose()
    plt.figure()
    set_title_and_hide_axis('Face grid')
#     print(result_image.shape)
    plt.imshow(result_image)

def show_extraction_results(face_features):
    face_features = np.asarray(face_features)
    plt.figure(figsize=(10,10))
    face_image, eye_images, face_grid = face_features
    plt.figure()
    set_title_and_hide_axis('Extracted face image')
    plt.imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB), interpolation="bicubic")
    plt.figure()
    render_face_grid(face_grid)

    for eye_image in eye_images:
        plt.figure()

        #print('eye image after extraction')
        set_title_and_hide_axis('Extracted eye image')
        plt.imshow(cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB), interpolation="bicubic")