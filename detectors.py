import cv2
import os
import numpy as np
from collections import namedtuple


OPEN_CV_DETECTOR_CONFIG = namedtuple("OPEN_CV_DETECTOR_CONFIG", ["FACE_DETECTOR_FILE_PATH", "EYE_DETECTOR_FILE_PATH"])


class OpenCvDetector(object):
    def __init__(self, config):
        self.face_detector = cv2.CascadeClassifier(config.FACE_DETECTOR_FILE_PATH)
        self.eye_detector = cv2.CascadeClassifier(config.EYE_DETECTOR_FILE_PATH)

    def detect_faces(self, image, scale_factor, min_neighbors):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scale_factor, min_neighbors)
        return faces

    def detect_eyes(self, image, scale_factor, min_neighbors):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        eyes = self.eye_detector.detectMultiScale(gray, scale_factor, min_neighbors)
        return eyes

class OpenCvDetector(object):
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier(os.path.join(".", "detector_architectures",
                                                                "haarcascade_frontalface_default.xml"))
        self.eye_detector = cv2.CascadeClassifier(os.path.join(".", "detector_architectures",
                                                               "haarcascade_eye.xml"))

    def detect_faces(self, image, scale_factor, min_neighbors):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scale_factor, min_neighbors)
        print("{} faces detected".format(len(faces)))
        return faces

    def detect_eyes(self, image, scale_factor, min_neighbors):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        eyes = self.eye_detector.detectMultiScale(gray, scale_factor, min_neighbors)
        print("{} eyes detected".format(len(eyes)))
        return eyes


class FaceKeyPointDetector(object):
    def __init__(self, model, face_detector):
        self.model = model
        self.face_detector = face_detector

    def _prepare_extracted_faces(self, image, face_boundaries):
        (x, y, w, h) = face_boundaries
        face_img = image[y:y+h, x:x+w, :]

        face_reshaped = cv2.resize(face_img, (96, 96))
        gray = cv2.cvtColor(face_reshaped, cv2.COLOR_RGB2GRAY)

        gray_normalized = gray / 255.
        gray_normalized = gray_normalized[np.newaxis, :, :, np.newaxis]
        return gray_normalized

    def _re_normalize_key_points(self, key_points, face_boundaries):
        (x, y, w, h) = face_boundaries
        key_points = key_points * 48 + 48

        x_coords = key_points[0][0::2]
        y_coords = key_points[0][1::2]

        x_coords = x_coords * w / 96 + x
        y_coords = y_coords * h / 96 + y
        return x_coords, y_coords

    def detect(self, sess, image, scale_factor, min_neighbors):
        face_boundaries = self.face_detector.detect_faces(image, scale_factor, min_neighbors)
        face_keypoints = []
        for face in face_boundaries:
            normalized_face_gray = self._prepare_extracted_faces(image, face)
            key_points = self.model.predict(sess, normalized_face_gray)
            x_coords, y_coords = self._re_normalize_key_points(key_points, face)
            face_keypoints.append((x_coords, y_coords))
        return face_keypoints