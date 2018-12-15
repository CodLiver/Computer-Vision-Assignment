################################################################################

# functionality: perform detection based on HOG feature descriptor / SVM classification
# using a very basic multi-scale, sliding window (exhaustive search) approach

# This version: (c) 2018 Toby Breckon, Dept. Computer Science, Durham University, UK
# License: MIT License

# Minor portions: based on fork from https://github.com/nextgensparx/PyBOW

################################################################################

import cv2
import os
import numpy as np
import math
import hogClassifier.params as params
from hogClassifier.utils import *
from PIL import ImageFont, ImageDraw, Image
################################################################################

show_scan_window_process = False;

################################################################################

# load SVM from file

class HOGdetector():
    """docstring for HOGdetector."""
    def __init__(self,path):#, PATH_Train
        super(HOGdetector, self).__init__()
        #self.PATH_Train = PATH_Train
        try:
            self.svm = cv2.ml.SVM_load("./hogClassifier/"+path+".xml")#params.HOG_SVM_PATH
        except:
            print("Missing files - SVM!");
            print("-- have you performed training to produce these files ?");
            exit();
        # print some checks
        print("svm size : ", len(self.svm.getSupportVectors()))
        print("svm var count : ", self.svm.getVarCount())

        # self.ss=Selective_Search()

    def detect(self,frame):
        "hog object detection"
        img=frame
        output_img = False#

        # for a range of different image scales in an image pyramid
        detections = []

        ################################ for each re-scale of the image
        # for each window region get the BoW feature point descriptors

        img_data = ImageData(img)
        img_data.compute_hog_descriptor();

        # generate and classify each window by constructing a BoW
        # histogram and passing it through the SVM classifier

        if img_data.hog_descriptor is not None:

            #print("detecting with SVM ...")

            retval, [result] = self.svm.predict(np.float32([img_data.hog_descriptor]))

            # if we get a detection, then record it
            if result[0] == 1.0:
                rect = np.float32([0, 0, frame.shape[0], frame.shape[1]])
                output_img=True
        ########################################################


        return output_img
#####################################################################
