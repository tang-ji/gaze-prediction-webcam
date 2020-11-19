import dlib
import numpy as np
import matplotlib.pyplot as plt

class LandmarksDetector:
    def __init__(self, predictor_model_path):
        self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):
        dets = self.detector(image, 1)
        ret = []
        for detection in dets:
            ret += [[item.x, item.y] for item in self.shape_predictor(image, detection).parts()]
        return np.array(ret)
    
landmarks_detector = LandmarksDetector("model/shape_predictor_5_face_landmarks.dat")

def draw_image(image):
    _, ax = plt.subplots(figsize=(8,8), dpi=80)
    ax.imshow(image)
    ax.set_yticks([])
    ax.set_xticks([])

def get_draw_prediction_directory(image):
    preds = landmarks_detector.get_landmarks(image)
    _, ax = plt.subplots(figsize=(8,8), dpi=80)
    ax.imshow(image)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.scatter(x=preds[:, 0], y=preds[:, 1], c='w', s=10)
    return preds
