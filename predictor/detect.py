import sys
import time
from PIL import Image, ImageDraw
#from models.tiny_yolo import TinyYoloNet
from predictor.utils import *
from predictor.image import letterbox_image, correct_yolo_boxes
from predictor.darknet import Darknet



class detectorClass():
    """docstring for detectorClass."""
    def __init__(self, cfgfile, weightfile):
        super(detectorClass, self).__init__()
        self.weightfile = weightfile
        self.cfgfile = cfgfile
        self.m = Darknet(self.cfgfile)
        #self.m.print_network()
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

        # task is to find the returners of plot_boxes()
        # return boxes,class_names,etc

    def detect_cv2(self, imgfile, depthMx):#save
        "probably I will use this?"
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


###modularize this part, remove unnecessary parts. output the rectangle areas for the DL deep learning.
##possible I/O structure.
# predefined weight class definition.
# I= either whole or partial image. lets go with whole
# deploy this to detect(), wfile will be pre alloced by class.
# return array [[label,h,w]]
# disparity will get it and compare with its own stuff. get the result.
if __name__ == '__main__':
    dc=detectorClass("cfg/yolo_v3.cfg", "yolov3.weights")
    print("started")
    dc.detect_cv2("data/1506942477.481815_L.png",'data/test1.jpg')
    print("end")





    # if len(sys.argv) == 5:
    #     cfgfile = sys.argv[1]
    #     weightfile = sys.argv[2]
    #     imgfile = sys.argv[3]
    #     globals()["namesfile"] = sys.argv[4]
    #     detect(cfgfile, weightfile, imgfile)
    #     #detect_cv2(cfgfile, weightfile, imgfile)
    #     #detect_skimage(cfgfile, weightfile, imgfile)
    # else:
    #     print('Usage: ')
    #     print('  python detect.py cfgfile weightfile imgfile names')
    #     #detect('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg', version=1)
