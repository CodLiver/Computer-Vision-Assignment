# OpenCVassignment
just scared of losing it.


### General ToDo

DONE - go with NN that,

DONE/Depr - use AOI: front(cars,pedestrians), right(cars), left(pedestrians). more classes if want more?

DONE/Depr - filter out
    DONE - compare/deploy to DL with RGB. If fits, then sits.
    'maybe use SVM?'

DONE/Depr - Train pedestrians with DL. If your implementation catches area, check with DL and give the result.
    Deprecated 'very bad results, may use SVM, add cars dataset to yours, retrain check, different sizes dataset.'
    DONE WITH YOLOOO

Semi-DONE - distance: use color map of disparity to calculate distance. If model says this, distance be taken, pipelined to boxing
DONE BY YOLOOO - boxing: if yes, box the area. maybe HOG? maybe own heatmap?

CURRENT OBSERVATIONS:

dl is bad. use HOG instead, personas are seen as cars or literally anything else.

NVM, YOLOOO solved it all!!!

now set the class get all detections here. do optimization where needed. Insert numbers with the coords given.


Object Detection with YOLOv3 DONE.

-----------------

### Supporting tests:

Sliding Window: GOOD

Selective Search: BAD

update dataset

remove tests and put into train

adapt the hog_train and params to multi purpose:

Train on mira

test on disparity again.

---------------

### Report:

write report, write whatever he says

compare sliding window with selective search

talk about your application neural network too

compare HOG vs BOW vs NN vs YOLOv3

acknowledge prev. datasets


Record Video




resources

https://pjreddie.com/media/files/papers/YOLOv3.pdf
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
    }
