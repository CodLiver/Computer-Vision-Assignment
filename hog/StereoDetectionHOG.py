#####################################################################

# Example : load, display and compute SGBM disparity
# for a set of rectified stereo images from a  directory structure
# of left-images / right-images with filesname DATE_TIME_STAMP_{L|R}.png

# basic illustrative python script for use with provided stereo datasets

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2017 Department of Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

#!/usr/bin/python
"""Cem"""
from matplotlib import pyplot as plt
import time,cv2,os,sys
# print(sys.path)
# from Classifier import Classifier
# cl=Classifier("D:/ComputerVisionAssignment/results/","D:/ComputerVisionAssignment/newTrainedModel.pth")
import numpy as np
# from predictor.detect import *
from hog_detector import HOGdetector
from selective_search import Selective_Search

"""
PLAN:

* Check selective search
* retrain with cars
* improve preprocessing
* start report
* start video

"""




# import predictor

hd=HOGdetector()
# sss=Selective_Search()

# where is the data ? - set this to where you have it

master_path_to_dataset = "D:/ComputerVisionAssignment/TTBB-durham-02-10-17-sub10/"; # ** need to edit this **
directory_to_cycle_left = "left-images";     # edit this if needed
directory_to_cycle_right = "right-images";   # edit this if needed

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres
fB=stereo_camera_baseline_m*camera_focal_length_px

# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns

skip_forward_file_pattern = ""; # set to timestamp to skip forward to

crop_disparity = False; # display full or cropped disparity image
pause_playback = False; # pause until key press after each image

#####################################################################

# resolve full directory location of data set for left / right images

full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);

# get a list of the left image files and sort them (by timestamp in filename)

left_file_list = sorted(os.listdir(full_path_directory_left));

# setup the disparity stereo processor to find a maximum of 128 disparity values
# (adjust parameters if needed - this will effect speed to processing)

# uses a modified H. Hirschmuller algorithm [Hirschmuller, 2008] that differs (see opencv manual)
# parameters can be adjusted, current ones from [Hamilton / Breckon et al. 2013]

# FROM manual: stereoProcessor = cv2.StereoSGBM(numDisparities=128, SADWindowSize=21);

# From help(cv2): StereoBM_create(...)
#        StereoBM_create([, numDisparities[, blockSize]]) -> retval
#
#    StereoSGBM_create(...)
#        StereoSGBM_create(minDisparity, numDisparities, blockSize[, P1[, P2[,
# disp12MaxDiff[, preFilterCap[, uniquenessRatio[, speckleWindowSize[, speckleRange[, mode]]]]]]]]) -> retval

max_disparity = 128;
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);


BWValue=[]
counter=0
# plt.ion()
# plt.show()
for filename_left in left_file_list:
    # skip forward to start a file we specify by timestamp (if this is set)

    if ((len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename_left)):
        continue;
    elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left)):
        skip_forward_file_pattern = "";

    # from the left image filename get the correspondoning right image

    filename_right = filename_left.replace("_L", "_R");
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left);
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right);

    # for sanity print out these filenames

    print(full_path_filename_left);
    print(full_path_filename_right);
    print();

    # check the file is a PNG file (left) and check a correspondoning right image
    # actually exists

    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel
        # RGB images so load both as such

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)#CEM )
        # cv2.imshow('left image',imgL) #I REMOVED

        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
        "smoothing was tried to enhance the quality, it didnt go well."
        # cv2.imshow('right image',imgR) #I REMOVED

        print("-- files loaded successfully");
        print();

        # remember to convert to grayscale (as the disparity matching works on grayscale)
        # N.B. need to do for both as both are 3-channel images

        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY);
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY);

        # perform preprocessing - raise to the power, as this subjectively appears
        # to improve subsequent disparity calculation

        grayL = np.power(grayL, 0.75).astype('uint8');
        grayR = np.power(grayR, 0.75).astype('uint8');

        # compute disparity image from undistorted and rectified stereo images
        # that we have loaded
        # (which for reasons best known to the OpenCV developers is returned scaled by 16)

        disparity = stereoProcessor.compute(grayL,grayR);

        # filter out noise and speckles (adjust parameters as needed)

        dispNoiseFilter = 5; # increase for more agressive filtering
        cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);

        # scale the disparity to 8-bit for viewing
        # divide by 16 and convert to 8-bit image (then range of values should
        # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
        # so we fix this also using a initial threshold between 0 and max_disparity
        # as disparity=-1 means no disparity available

        _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO);

        """

        ####################################### CEM'S PART START ####################################################
		ask this part to change:

		DONE - go with that,

		DONE - use AOI: front(cars,pedestrians), right(cars), left(pedestrians). more classes if want more?

		DONE - filter out
        DONE - compare/deploy to DL with RGB. If fits, then sits.
        'maybe use SVM?'

		DONE - Train pedestrians with DL. If your implementation catches area, check with DL and give the result.
        Deprecated 'very bad results, may use SVM, add cars dataset to yours, retrain check, different sizes dataset.'
        DONE WITH YOLOOO

        Semi-DONE - distance: use color map of disparity to calculate distance. If model says this, distance be taken, pipelined to boxing
        DONE BY YOLOOO - boxing: if yes, box the area. maybe HOG? maybe own heatmap?

        CURRENT OBSERVATIONS:
        dl is bad. use HOG instead, personas are seen as cars or literally anything else.
        NVM, YOLOOO solved it all!!!
        now set the class get all detections here. do optimization where needed. Insert numbers with the coords given.

        https://pjreddie.com/media/files/papers/YOLOv3.pdf
        @article{yolov3,
          title={YOLOv3: An Incremental Improvement},
          author={Redmon, Joseph and Farhadi, Ali},
          journal = {arXiv},
          year={2018}
        }
        """

        "#THE DEPTH CALC MATRIX, BUT MIGHT BE WRONG, LET PEOPLE CHECK"
        # disparity=cv2.resize(disparity[:,124:],(1024,544))
        # depthMx=fB/disparity
        # depthMx[depthMx>255]=0.4

        disparity_scaled = (disparity / 16.).astype(np.uint8);

        RESULT=(disparity_scaled * (256. / max_disparity)).astype(np.uint8)[20:420,124:]

        cv2.imshow("disparity",RESULT);

        LHS=RESULT[:,:200]
        RHS=RESULT[:,600:]
        MAIN=RESULT[:,200:600]
        rightFiltered=cv2.inRange(RHS, 40, 100)[:300,:]
        mainFiltered=cv2.inRange(MAIN, 20, 40)[:245,:380]
        leftFiltered=cv2.inRange(LHS, 40, 100)[:320]
        countedRight=cv2.countNonZero(rightFiltered)
        countedMain=cv2.countNonZero(mainFiltered)
        countedLeft=cv2.countNonZero(leftFiltered)

        imgRR=imgL[:,600:]
        imgMM=imgL[:,200:600]
        imgLL=imgL[:,:200]

        if countedRight>5000:
            #cv2.imwrite("D:/ComputerVisionAssignment/results/"+str(counter)+"r.png",imgL[:,600:])#[:300,:])
            # cv2.imshow("processedR", hd.detect(imgL[:,600:]))
            # imgRR=hd.detect(imgL[:,600:])
            imgRR=hd.detectSelective(imgL[:,600:])
            # imgRR=sss.detect(imgL[:,600:])


        if countedMain<50000 and countedMain>10000:
            # cv2.imwrite("D:/ComputerVisionAssignment/results/"+str(counter)+"m.png",imgL[:,200:600])#[:245,:400])
            # cv2.imshow("processedM", hd.detect(imgL[:,200:600]))
            # imgMM=hd.detect(imgL[:,200:600])
            imgMM=hd.detectSelective(imgL[:,200:600])
            # imgMM=sss.detect(imgL[:,200:600])

        if countedLeft<27000 and countedLeft>7000:
            # cv2.imwrite("D:/ComputerVisionAssignment/results/"+str(counter)+"l.png",imgL[:,:200])#[:320])#may change the second ones
            # cv2.imshow("processedL", hd.detect(imgL[:,:200]))
            # imgLL=hd.detect(imgL[:,:200])
            imgLL=hd.detectSelective(imgL[:,:200])
            # imgLL=sss.detect(imgL[:,:200])


        imMR=np.concatenate((imgMM, imgRR), axis=1)
        imLMR=np.concatenate((imgLL, imMR), axis=1)

        cv2.imshow("processed",imLMR)




        """seperation done, start histogram them, observe change, use either HOG or DL"""

        """####################################### CEM'S PART END ####################################################"""



        # keyboard input for exit (as standard), save disparity and cropping
        # exit - x
        # save - s
        # crop - c
        # pause - space

        key = cv2.waitKey(40 * (not(pause_playback))) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if (key == ord('x')):       # exit
            break; # exit
        elif (key == ord('s')):     # save
            cv2.imwrite("sgbm-disparty.png", disparity_scaled);
            cv2.imwrite("left.png", imgL);
            cv2.imwrite("right.png", imgR);
        elif (key == ord('c')):     # crop
            crop_disparity = not(crop_disparity);
        elif (key == ord(' ')):     # pause (on next frame)
            pause_playback = not(pause_playback);
    else:
            print("-- files skipped (perhaps one is missing or not PNG)");
            print();

# close all windows

cv2.destroyAllWindows()

#####################################################################
