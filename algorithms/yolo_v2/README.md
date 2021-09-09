[image1]: ./images/yolov2_loss_notes.jpg "YOLOv2 Loss"
[image2]: ./images/yolov2_160e_darknet150e_charts.png "YOLOv2 Training Charts"

# YOLO_v2 implementation
Implemention of YOLO_v2 from scratch in PyTorch (1.8.1). 

Done for learning purposes, in order to get a better understanding of PyTorch and the ins and outs of YOLO algorithm. Shared for reference, along with the evaluation results and some coding tips. 

To make the code more clear and understandable without extra knowledge - I intentionally avoid using helper classes, functions etc. The result is less compact code but hopefully easier to digest without having to skip from one script to another in order to figure out what this function or that class does.

---

## Key Files
- **models:**
	- model.py - YOLOv2 model
	- darknet_model.py - Darknet19 backbone (first 19 conv layers of YOLO model that are pre-trained on ImageNet for classification task)
- **training**
	- train.py - YOLO_v1 training (use this to train YOLO without Darknet pretraining or after running darknet_train)
	- darknet_train.py - Darknet pre-training on ImageNet. Single GPU/CPU version (used this just for quick testing /debugging). After it's done, saves the model into .pt file that can be loded in train.py before training for detection.
	- darknet_train_multigpu_distr.py - Darknet pre-training on ImageNet. Multi GPU version (this is what I used most to train on full ImageNet).
	- darknet_train_multigpu.py - Darknet pre-training on ImageNet.  Multi GPU version that only distributes data with DataParallel. Doesn't speed up much/at all, just allows to train bigger batch sizes.
- **test.py** - testing script to calculate mAP for specified test dataset and using some pre-trained YOLOv2 model
- **loss.py** - YOLOv2 loss calculation
- **dataset.py** - PyTorch Dataset implementation for Pascal VOC
- **detect_txt.py** - generate predictions in txt files corresponding to each image file (similar as ground truth). This is used for calculating the final mAP with Evaluation toolkit by Padilla (linked at the bottom of this page).
- **detect.py** - detect bouding boxes and plot them onto images, using pre-trained YOLOv2 model.
- **utils.py** - some helper functions borrowed from YOLOv1 implementation of Aladdin Persson (bounding box extraction, mAP calculation etc.)

## Datasets used
- Darknet19 training/testing - [ImageNet](https://image-net.org/) (ILSVRC2014 Competition Dataset with 1000k classes)
- YOLOv2 training: [PascalVOC](http://host.robots.ox.ac.uk/pascal/VOC/]) 2007+2012 | testing: Pascal VOC2007

---

## Overview

Implementation was mostly done based on YOLO papers. I used v1, v2 and v3 papers. 

V1 paper becasue some things are unchanged in v2; v3 paper was used as some new architecture bits are ommited in v2 paper and yet described in more detail in v3 one, e.g. anchors. 

To start off, I used my past YOLOv1 code and modyfied it to fit whatever changed in v2 (e.g. I moved over bits of utils.py by Aladdin Persson from his excelent YOLOv1 tutorial).

**Similar as with my YOLOv1 implementation:**
- Darknet model/training/testing are seperated from YOLOv2 model. YOLOv2 can be trained from scratch or starting with weights for first 19 conv layers loaded from file saved by darknet training script.
- I trained Darknet19 on ImageNet and the results of that can be seen below.
- YOLOv2 was trained on same Pascal VOC datset as v1 (same like in the YOLOv2 paper, except I didn't follow up on the additional MSCOCO training/testing they did there).

**Key differences to YOLOv1 (from coding/implementation perspective):**
- the input dataset format is same as in YOLOv1, howver dataset.py is updated as input needs to be transformed in a new way to fit the new YOLOv2 architecture and updated loss function:
	- previously in YOLO v1, we had 2 bounding boxes predicted and one box per cell in ground truth. In the loss function, we picked just one predicted box for error calculation vs. ground truth (pick was based on which has better IoU with ground truth); 
	- in YOLO v2, there are 5 predicted bounding boxes per cell, relative to 5 possible boxes in ground truth. Loss compares predicted coords of those 5 boxes to each ground truth.
	- in both cases (v1 and v2) for cells where ground truth doesn't exist the identity function takes zero (basically a multipllier that zeroes out the error)
	- in YOLO v1 all coords were directly predicted by network, in v2 the predictions are an offset from predefined widths/heights (called "anchors" or "priors"). During loss calculation the predictions coming from network are transformed before comparing to ground truth (Fig. 3 in YOLOv2 paper).
	- the mentioned anchors are defined by box widths and heights. In the original Darknet implementation/ YOLOv2 paper those are obtained seperatly and prior to network training using k-means algorithm on groud truth boxes of training set. In this implementation, I copied over the anchors from original YOLOv2 implementaiton config file ([yolov2-voc.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-voc.cfg))
- the components of loss function "on paper" are not that much different to v1, however differences in network output and tensor format for data sample require to modify the loss code in comparison to YOLOv1. Likewise, the inclusion of anchors also needs to be reflected in loss code (see equations in page 3 of YOLOv2 paper, b_x, b_y, b_w, b_h, also some of my notes below)
- earlier implementation of YOLO had only resize image transformations, this YOLOv2 uses more augmentations and therefore added new external library albumentations (torchvision has auugmentation support but mostly for classification, ie. doesnt modify bounding boxes)
- for supporting functions in utils.py in comparison to YOLO v1, I changed the non-max surpression implementation. Turns out the previous one I used was quite slow. Now I switched to torchvision implementation that's multifold times faster. This helps to debug and track the accuracy during training (nms needs to be applied in order to calculate mAP)

![alt text][image1]
*Fig. 1 YOLOv2 Loss with my notes. The equation found at [towardsdatascience](https://towardsdatascience.com/training-object-detection-yolov2-from-scratch-using-cyclic-learning-rates-b3364f7e4755) (it's same as one in YOLOv1 paper, just slighty different notation), the bbox image and variable definitions are from YOLOv2 paper.*


**Notes for source code (same as YOLOv1):**
- I implemented model, training, testing, detection, loss and dataset scripts. The helper functions in utils are taken directy from mentioned tutorial with some modifications to suit the new YOLOv2 architecture. 
- The mAP calculated in training/test scripts is only for hardcoded 0.4 confidence threshold and 0.5 IoU. Note, this is not the metric used by Pascal VOC and/or typically reported in research papers. 
- To obtain the Pascal VOC mAP, use evaluation toolkit by Padilla et al. The Pascal VOC mAP metric calculates precision and recall (for every class seperatly) at all confidence levels. Subsequently, precision is sampled by taking max for all/selected recall levels in order to calculate AP (avg across all sampled precisions). The processss is explained quite well [here](https://datascience.stackexchange.com/questions/25119/how-to-calculate-map-for-detection-task-for-the-pascal-voc-challenge). 
- In practice, all above this means that during train/test the displayed mAP will usually be quite a bit lower than reference found in published YOLO paper, as it is only calculated on one fixed confidence level. The final Pascal VOC metric takes into account best confidence scenario that maxes out precision.

**Some of my implementation experiences specific for YOLOv2 (that might help if you're trying to redo this on your own):**
- The parts that took most work are dataset.py and loss.py. Later, also modyfing some old code YOLOv1 in utils.py to match v2 output and architecture (bbox extraction from network output etc.). 
- Debugging-wise biggest time sink was the loss function (I'd say \~90%). I had multiple bugs that were quite hard to spot at first glance (e.g. applied sigmoid twice to same tensor; or skipped sigmoid where it was needed; reshaped the network output wrongy). The difficulty in debugging was that before and after removing the bugs, network behaviour was not all that much different (ie. even with the bugs loss was going down as it should, no odd numbers etc.).
- The new network architecture is relativly straightfoward to code and I don't think I had any bugs in it at all. At some point when I couldn't find any bugs but network was not properly learning - I disabled the skip connection to make network simpler (later on turned out that was working quite ok).
- Things that eventually helped me track down the issues: 1) display all loss components and see how they change as epochs go by (that helped me figure out that my obj and non-obj loss were behaving different and tracked that one had a bug); 2) insert one very simple example into the network, calculate manually all intermediate loss values and components. Debug, literally, line by line and compare values to manually calculated ones (long and tedious but in the end solved pretty much all problems).
- when debugging remember to turn off augmentations (I kept forgetting this too many times)
- learning rate modifications seem to have quite some impact on results when training/testing small datasets
- for single sample you should be able to obtain close to 0.9 accuracy just with few training epochs (like \~10 epochs, later testing on same sample), for 100 samples takes a bit longer (\~30 epoch)


## Results

In short, when using pre-trained Darknet19 on ImageNet and later training on VOC2007+2012 I managed to obtain mAP 0.72 testing on VOC2007 (calculated with Padilla et al. scripts). 

Same model but testing with a fixed 0.4 confidence threshold this corresponds to mAP 0.91 on train set and 0.56 on test. For Darknet19 pre-training, I got 91.6% top-5 accuracy and 74.0% top-1 accuracy on ImageNet test set after 150 epochs of training on ImageNet train set.


![alt text][image2]
*Fig. 2 YOLOv2 training charts: red = constant LR (2e-5), 180 epochs and loss weights set (2); blue = constant LR (2e-5) and loss weights set (1); green = scheduled LR).*

## Notes for running the code (similar as YOLOv1)
- configuration parameters for each script (train/test/detect etc.) are global variables at the top of the source code (paths to datasets, pre-trained models; hyperparameters etc.)
- first epoch always takes a bit longer to train
- all paths in train/test/detect scripts need to be updated to match your dataset directories. 
- this repository does not include the full datasets required to train the model, however I included tiny samples for both Pascal VOC and ImageNet to test things out more smoothly (look in the YOLO_v1 directory)
- in the evaluation table, I note the times and performances for multiple runs as I used a shared remote server that allowed running code only in certain blocks of time (e.g. 24h, 48h etc.)
- in comparison to the YOLOv2 paper, my final version doesn't include Darknet training with the high resolution images, nor the YOLOv2 training with multi-scale (ie. switching scale every 10 epochs); I tested those solutions but didn't give me particularly improved results
- LR setting can make quite a difference. The setting from the paper for scheduled LR decreasing by e-10 didn't work that good for me, after playing around with some other LR set I managed to improve it. However, in the end constant LR worked best.
- weights for loss components do make a difference as well. My initial weights had greatly elevated object loss and then box loss (set 1), when I made all weights closer to each other just slightly favouring object loss mAP went up by 3% (set 2) ! 

### Darknet19 backbone training results (classification)

**MULTI-GPU (4x V100 , batch size: 64 , starting lr: 1 , with decay exponent 0.1, ie. ends on lr 0.28, when doing 150 epochs )**

|Run|Epoch start|Epoch end|Time (h)|Val Acc1 (%)|Val Acc5 (%)|Train Loss|comments |
|---|-----------|---------|--------|------------|------------|----------|---------|
|1  |0          |29       |35      |18.3        |40.1        |4.14      |         |
|2  |30         |59       |35      |32.6        |59.2        |3.27      |         |
|3  |60         |89       |35      |58.8        |82.4        |2.18      |         |
|4  |90         |119      |32      |72.4        |90.8        |1.32      |         |
|5  |120        |149      |33      |74.0        |91.6        |1.13      |         |


---

**Sources:**
- YOLO v1 paper by Joseph Redmon et al.: ["You Only Look Once: unified, Real-Time Object Detection"](https://arxiv.org/abs/1506.02640)
- YOLO v2 paper by Joseph Redmon et al.: ["YOLO9000: Better, Faster, Stronger"](https://arxiv.org/abs/1612.08242)
- YOLO v3 paper by Joseph Redmon et al.: ["YOLOv3: An Incremental Improvement"](https://arxiv.org/abs/1804.02767)
- YOLO v1 coding tutorial by Aladdin Persson: [Youtube](https://www.youtube.com/watch?v=n9_XyCGr-MI]) | [Git](https://github.com/aladdinpersson/Machine-Learning-Collection)
- Evaluation toolkit by Padilla et al.: ["A Comparative Analysis of Object Detection Metrics with a Companion Open-Source Toolkit"](https://github.com/rafaelpadilla/Object-Detection-Metrics)