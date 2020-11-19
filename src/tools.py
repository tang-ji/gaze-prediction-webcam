import dlib, cv2
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

def draw_image(image, t=False):
    _, ax = plt.subplots(figsize=(8,8), dpi=80)
    if t:
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(image)
    ax.set_yticks([])
    ax.set_xticks([])

def get_bound(points):
    x_max = np.max(points, axis=0)[0]
    x_min = np.min(points, axis=0)[0]
    y_max = np.max(points, axis=0)[1]
    y_min = np.min(points, axis=0)[1]
    x_d = (x_max-x_min)*0.1
    x_min -= x_d
    x_max += x_d
    y_d = (y_max-y_min)*0.3
    y_min -= y_d
    y_max -= y_d
    return int(x_max), int(x_min), int(y_max), int(y_min)

def get_box(bound):
    x_max, x_min, y_max, y_min = bound
    return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

def warpBox(image,
            box,
            target_height=None,
            target_width=None,
            return_transform=False):
    """Warp a boxed region in an image given by a set of four points into
    a rectangle with a specified width and height. Useful for taking crops
    of distorted or rotated text.
    Args:
        image: The image from which to take the box
        box: A list of four points starting in the top left
            corner and moving clockwise.
        target_height: The height of the output rectangle
        target_width: The width of the output rectangle
        return_transform: Whether to return the transformation
            matrix with the image.
    """
    box = np.float32(box)
    w, h = image.shape[1], image.shape[0]
    assert (
        (target_width is None and target_height is None)
        or (target_width is not None and target_height is not None)), \
            'Either both or neither of target width and height must be provided.'
    if target_width is None and target_height is None:
        target_width = w
        target_height = h
    M = cv2.getPerspectiveTransform(src=box, dst=np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]]).astype('float32'))
    full = cv2.warpPerspective(image, M, dsize=(int(target_width), int(target_height)))
    if return_transform:
        return full, M
    return full

