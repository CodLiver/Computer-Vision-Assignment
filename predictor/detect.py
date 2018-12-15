import sys
import time,os
from PIL import Image, ImageDraw
#from models.tiny_yolo import TinyYoloNet
from predictor.utils import *
from predictor.image import letterbox_image, correct_yolo_boxes
from predictor.darknet import Darknet

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

class detectorClass():
    """
    PyTorch Implementation of YOLOv3 Detector class:

    detect.py modularization and some assignment related integrations done by me, but the owner of the whole "predictor" folder (incl. other files and subfiles) is:

    /***************************************************************************************
    *    Title: pytorch-0.4-yolov3
    *    Author: Young-Sun (Andy) Yun (GitHub: andy-yun)
    *    Date: June 1 2018
    *    Code version: 0.9.1
    *    Availability: https://github.com/andy-yun/pytorch-0.4-yolov3
    *
    ***************************************************************************************/

    __Methods that are used__

    __init__(cfgfile, weightfile):
    initalize by entering the relative path of 'cfgfile' in cfg file and 'weightfile' under 'predictor' which are already done in StereoDetectionYOLOv3.py

    detect_cv2(img, depthMatrix):
    ->
    INPUTS:
    img: 'numpy array' or 'path to file'.
    depthMatrix: disparity matrix where closer images are dark and distant images are brighter (each pixel is calculated as meter.)
    <-
    OUTPUTS:
    img=boxed image with distances to objects.
    nameAndCoord= array that contains the image's label and distance (if only commandline was used.)

    this method was used in the assignment. Please do not use other methods, as they might break the algorithm.


    """
    def __init__(self, cfgfile, weightfile):
        super(detectorClass, self).__init__()
        self.weightfile = weightfile
        self.cfgfile = cfgfile
        self.m = Darknet(self.cfgfile)
        self.m.load_weights(self.weightfile)
        print('Loading weights from %s... Done!' % (self.weightfile))

        if self.m.num_classes == 20:
            namesfile = 'data/voc.names'
        elif self.m.num_classes == 80:
            self.namesfile = 'predictor/data/coco.names'
        else:
            self.namesfile = 'data/names'
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.m.cuda()

    def detect(self, imgfile, save):

        # import cv2
        img = Image.open(imgfile).convert('RGB')

        # returns resized img with good aspects that I am not sure what. is simple resize helps?
        sized = letterbox_image(img, self.m.width, self.m.height)
        start = time.time()

        "gets all best prediction boxes. predictions done here. Model is quite taking time apprx >1sec."
        boxes = do_detect(self.m, sized, 0.5, 0.4, self.use_cuda)

        "sizes are corrected to images size"
        correct_yolo_boxes(boxes, img.width, img.height, self.m.width, self.m.height)

        finish = time.time()
        print('Predicted in %f seconds.' % (imgfile, (finish-start)))

        class_names = load_class_names(self.namesfile)
        plot_boxes(img, boxes, save, class_names)


    def detect_cv2(self, imgfile, depthMx):
        "detect_cv2, the method that is used for detection."
        import cv2

        if type(imgfile) is np.ndarray:
            img = imgfile
        else:
            img = cv2.imread(imgfile)
        sized = cv2.resize(img, (self.m.width, self.m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        for i in range(2):
            start = time.time()
            boxes = do_detect(self.m, sized, 0.5, 0.4, self.use_cuda)
            finish = time.time()
            if i == 1:
                print('Predicted in %f seconds.' % (finish-start))

        class_names = load_class_names(self.namesfile)
        return plot_boxes_cv2(img, boxes, depthMx, class_names=class_names) #return added.  3rd param instead depthMx,savename=save

    def detect_skimage(self, imgfile, save):
        from skimage import io
        from skimage.transform import resize

        img = io.imread(imgfile)
        sized = resize(img, (self.m.width, self.m.height)) * 255

        for i in range(2):
            start = time.time()
            boxes = do_detect(self.m, sized, 0.5, 0.4, self.use_cuda)
            finish = time.time()
            if i == 1:
                print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

        class_names = load_class_names(self.namesfile)
        plot_boxes_cv2(img, boxes, savename=save, class_names=class_names)
