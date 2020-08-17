import cv2, time, pandas
from datetime import datetime
import os
import glob
import numpy as np

SIZE = (600, 400)
PROB_THRES = 0.35


class Human:
    def __init__(self,
                 output_path="output/replace.avi",
                 image_path="res/stick.png"):
        self.OUTPUT_PATH = output_path
        # read replacement image
        self.stick = cv2.imread(image_path, -1)

    def detect(self, video_path="input/parent-falling.mp4"):
        cap = cv2.VideoCapture(video_path)
        # add HOG + SVM detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # output writer
        out = cv2.VideoWriter(self.OUTPUT_PATH, cv2.VideoWriter_fourcc(*'DIVX'), 20, SIZE)

        n_skip = 2

        while True:
            # Capture frame-by-frame
            # skip 1 frame to speed up
            for _ in range(n_skip):
                _, _ = cap.read()
            grabbed, frame = cap.read()
            if not grabbed:
                break
            frame = cv2.resize(frame, SIZE)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # detect boxes and convert to numpy
            boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8))
            boxes, weights = np.asarray(boxes), np.asarray(weights)

            # filter boxes with low weight
            boxes = boxes[weights.flatten() > PROB_THRES]
            boxes = non_max_suppression_fast(boxes)

            new_frame = frame
            for box in boxes:
                #         (x, y, w, h) = box
                self.__overlay_img(new_frame, box)

            # save frame
            out.write(new_frame)

        # When everything done, release the capture
        out.release()
        cap.release()
        cv2.destroyAllWindows()

    # overlay image with stick figure
    def __overlay_img(self, image, dimensions):
        (x, y, w, h) = dimensions
        stick_cpy = self.stick.copy()
        stick_cpy = cv2.resize(stick_cpy, (w, h))

        alpha_stick = stick_cpy[:, :, 3] / 255.0
        alpha_img = 1.0 - alpha_stick
        blurred_back = image[y:y+h, x:x+w].copy()
        blurred_back = cv2.GaussianBlur(blurred_back, (21, 21), 0)

        for c in range(0, 3):
            image[y:y+h, x:x+w, c] = (alpha_stick * stick_cpy[:, :, c] +
                                      alpha_img * blurred_back[:, :, c])


# Based on https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlap_thresh=PROB_THRES):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int32")
