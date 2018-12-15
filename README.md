# OpenCVassignment


In this assignment, we got to implement an Human Detection Algorithm for driverless cars. The aim of this assignment is finding different approaches to human detection and classification. 

I have used Andy Yun's PyTorch implementation of YOLOv3 for object detection. I improved bounding boxes of detected objects when cv2 format images are processed. I integrated the Neural Network with the test dataset and managed classification. The success rate is more than 97%.

I also implemented/experimented several approaches like HOG and sliding window. As sliding window was exhaustively trying to search all possible places in the image, I cropped the image to have region of interest and reduced the boxes of detection. I used Sobel Edge Detection with respect to y-axis to reduce the unnecessary data and used opening with 3x3 kernel to remove the noise. I also converted human training dataset (INRIA Person dataset) to Sobel-Opening to increase the accuracy. Each time the candidate box passes the HOG detection, I also get the matrix coefficient of the box with random image from INRIA dataset to have another false positive filtering.

Also from the given disparity-image of the driverless car data, I created depth matrix to get the approximate distance between car and object.

Just download the program, change the dataset path in the python files. 



Original paper for YoloV3
https://pjreddie.com/media/files/papers/YOLOv3.pdf
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
 }


YoloV3 was taken from this page:
https://github.com/andy-yun/pytorch-0.4-yolov3

Stereo-disparity.py was implemented from:
https://github.com/tobybreckon/python-bow-hog-object-detection
