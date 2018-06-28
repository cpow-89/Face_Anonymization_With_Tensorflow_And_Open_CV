import cv2
import numpy as np


class FaceAnonymizer(object):
    def __init__(self, face_key_point_detector):
        self.face_key_point_detector = face_key_point_detector

    def _warp(self, image, dest_points):
        src_points = np.float32([[0, 0],
                                 [0, 499],
                                 [758, 499],
                                 [758, 0]])

        M = cv2.getPerspectiveTransform(src_points, dest_points)
        image_size = (image.shape[1], image.shape[0])
        warped = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)
        return warped

    def anonymize(self, sess, image, overlay_image, scale_factor, min_neighbors):

        output = np.copy(image)
        key_points = self.face_key_point_detector.detect(sess, image, scale_factor=scale_factor,
                                                         min_neighbors=min_neighbors)
        for person_points in key_points:
            important_points_x, important_points_y = person_points
            h = (important_points_y[5] - important_points_y[9]) * 3

            dest_pts = np.float32([[important_points_x[9], important_points_y[9]],
                                   [important_points_x[9], important_points_y[9] + h],
                                   [important_points_x[7], important_points_y[9] + h],
                                   [important_points_x[7], important_points_y[9]]])

            warped_overlay_image = self._warp(overlay_image, dest_pts)

            mask = warped_overlay_image[:, :, :3]
            mask[mask == 0] = 255

            output[mask != 255] = mask[mask != 255]

        return output
