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
"""mjqf76"""
from matplotlib import pyplot as plt
import time,cv2,os,sys
import numpy as np
import hogClassifier.classifier as HOG
from PIL import ImageFont, ImageDraw, Image

master_path_to_dataset = "TTBB-durham-02-10-17-sub10/"; # ** need to edit this **
displayAllowed=True #change if you run on mira without browsers

"""
mjqf76:

an algorithm that takes the bounding box ranges and performs HG detection for each bound.
for the candidate boxes, takes one training pedestrian image and takes the mean of their matrix coefficientself.
If they also correlate over certain threshold, then draw the rectangle and find the distance to car.
"""
def browser(counterx,countery,rectIncX,rectIncY,incX,incY,imgL,imgLCopy,sim,HOG1,depthMx,mins):
    while counterx<imgL.shape[1] and countery<imgL.shape[0]-272:
        while counterx<imgL.shape[1]:#for x should be full.
            "hog detection respect to sobel/opening training dataset."
            if HOG1.detect(imgL[countery:countery+int(incY/2),counterx:counterx+int(incX/2)]):
                "correlation with random pedestrian image to reduce additional false positives."
                try:
                    if(np.average(np.corrcoef(cv2.cvtColor(imgL[countery:countery+int(incY/2),counterx:counterx+int(incX/2)], cv2.COLOR_BGR2GRAY), sim)))>0.07:
                        imgLCopy=cv2.rectangle(imgLCopy,(counterx,countery),(counterx+rectIncX,countery+rectIncY),(0,255,0),3)
                        distance='%.1f' % (depthMx[countery:countery+int(incY/2),counterx:counterx+int(incX/2)].mean()*7)
                        if float(mins)>float(distance):
                        	mins=distance
                        # CV2 TO PIL #
                        "open cv has bad image draw, so PIL was used"
                        imgLCopy = cv2.cvtColor(imgLCopy, cv2.COLOR_BGR2RGB)
                        imgLCopy = Image.fromarray(imgLCopy)

                        imgLCopy.paste(Image.new('RGBA', (90,30), "black"), (counterx,countery))
                        ImageDraw.Draw(imgLCopy).text((counterx,countery), "H: "+str(distance)+"m", fill=(0,255,255,255), font=ImageFont.truetype("arialbd", 25))#
                        # # PIL TO CV2 back
                        imgLCopy = np.array(imgLCopy)
                        # # Convert RGB to BGR
                        imgLCopy = imgLCopy[:, :, ::-1].copy()
                except Exception as e:
                    pass

            counterx+=incX
        counterx=0
        countery+=incY
        return imgLCopy,mins





"an image that resembles the pedestrian most was chosen intuitively to make correlation with candidate boxes."
sim=cv2.imread("sim.png",0)
sim=cv2.resize(sim,(68,128))

"trained dataset is filtered with sobel/opening respect to y axis. "
HOG1=HOG.HOGdetector('svm_hogSOBEL')
# where is the data ? - set this to where you have it

directory_to_cycle_left = "left-images";     # edit this if needed
directory_to_cycle_right = "right-images";   # edit this if needed

"to find depth, stereo to 3d was used."
camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres
fB=stereo_camera_baseline_m*camera_focal_length_px

# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns

skip_forward_file_pattern = "";# set to timestamp to skip forward to
 #"1506943557.483956_L"  end of bailey
  #1506942940.478307_L start of bailey

crop_disparity = False; # display full or cropped disparity image
pause_playback = False; # pause until key press after each image

#####################################################################

# resolve full directory location of data set for left / right images

full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);

# get a list of the left image files and sort them (by timestamp in filename)

left_file_list = sorted(os.listdir(full_path_directory_left));

max_disparity = 128;
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);

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
    # check the file is a PNG file (left) and check a correspondoning right image
    # actually exists

    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel
        # RGB images so load both as such

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)#CEM )
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

        print("-- files loaded successfully");
        print();
        print(filename_left);
        # print(full_path_filename_right);


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
        """####################################### MJQF76'S PART START ####################################################"""
        "resize/preprocess disparity"
        disparity=cv2.resize(disparity[:,124:],(1024,544))
        depthMx=fB/disparity
        depthMx[depthMx>255]=0.4 #fill inf with vars.


        imgLCopy=imgL.copy()
        "sobel/opening pre-filtering to reduce noise and try to make the image recognizable"
        imgL=cv2.Sobel(imgR,cv2.CV_8U,1,0,ksize=3)
        imgL=cv2.morphologyEx(imgL, cv2.MORPH_OPEN, np.ones((1,1),np.uint8))

        "small scale sliding window that doesnt exhaustively search all the images, but the possible ones(AOI)."
        mins=40
        imgLCopy,mins=browser(0,128,136,256,136,256,imgL,imgLCopy,sim,HOG1,depthMx,mins)
        imgLCopy,mins=browser(68,128,136,256,136,256,imgL,imgLCopy,sim,HOG1,depthMx,mins)
        imgLCopy,mins=browser(0,128,68,128,136,256,imgL,imgLCopy,sim,HOG1,depthMx,mins)
        imgLCopy,mins=browser(34,128,68,128,136,256,imgL,imgLCopy,sim,HOG1,depthMx,mins)

        if displayAllowed:
            cv2.imshow("disparity",depthMx)
            cv2.imshow("detection",imgLCopy)

        if float(mins)<40:
            print(filename_right,"shortest detected image: Pedestrian(meter) ",mins);
        else:
            print(filename_right,"shortest detected image: No object has been detected!")
        print()

        """####################################### MJQF76'S PART END ####################################################"""



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
