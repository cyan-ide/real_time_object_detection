# YOLO_v1 implementation
Implemention of YOLO_v1 from scratch in PyTorch (1.8.1). 

Done for learning purposes, in order to get a better understanding of PyTorch and the ins and outs of YOLO algorithm. Shared for reference, along with the evaluation results and some coding tips. 

The key addition in comparison to other similar PyTorch implementations that can be found online is a script for training the Darknet backbone on ImageNet from scratch. 

This additional training takes a lot of time and requires multiple GPUs to complete in reasonable time so many often skip it. However, I do recommend implementing this yourself as well, as it teaches multiple aspects of PyTroch and netural nets related to scalability, evaluation, debugging and deployment.

---

## Key Files
- **models:**
	- model.py - YOLOv1 model
	- darknet_model.py - Darknet backbone (first 20 conv layers of YOLO model that are pre-trained on ImageNet for classification task)
- **training**
	- train.py - YOLO_v1 training (use this to train YOLO without Darknet pretraining or after running darknet_train)
	- darknet_train.py - Darknet pre-training on ImageNet. Single GPU/CPU version.
	- darknet_train_multigpu_distr.py - Darknet pre-training on ImageNet. Multi GPU version (this is what I used most to train on full ImageNet).
	- darknet_train_multigpu.py - Darknet pre-training on ImageNet.  Multi GPU version that only distributes data with DataParallel. Doesn't speed up much/at all, just allows to train bigger batch sizes.
- **test.py** - testing script
- **loss.py** - YOLO loss calculation
- **dataset.py** - PyTorch Dataset implementation for Pascal VOC
- **detect_txt.py** - generate predictions in txt files corresponding to each image file (similar as ground truth). This is used for calculating the final mAP with Evaluation toolkit by Padilla (linked above).

---

## Overview

For most part, when doing this implementation I followed the guidelines from an excelent tutorial on YOLO from Aladdin Persson (see link at bottom). However, going line by line I made some of my own modificataions and added tons of comments. 

Furthermore, the original tutorial ends only with training and validatoon on Pascal VOC train set, which proves that code runs but not much beyond that. Here, I go all the way til the end as in the original YOLO paper, my additions are: 
- seperated Darknet from YOLOv1
- pre-train Darknet backbone on ImageNet (I made single GPU / multi-GPU versions of training script)
- validate the final YOLO network on Pascal VOC validation dataset.

**Notes for source code:**
- I implemented model, training, testing, detection, loss and dataset scripts. The helper functions in utils are taken directy from mentioned tutorial without any modifications. 
- The mAP calculated in training/test scripts is only for hardcoded 0.4 confidence threshold and 0.5 IoU. Note, this is not the metric used by Pascal VOC and/or typically reported in research papers. 
- To obtain the Pascal VOC mAP, use evaluation toolkit by Padilla et al. The Pascal VOC mAP metric calculates precision and recall (for every class seperatly) at all confidence levels. Subsequently, precision is sampled by taking max for all/selected recall levels in order to calculate AP (avg across all sampled precisions). The processss is explained quite well [here](https://datascience.stackexchange.com/questions/25119/how-to-calculate-map-for-detection-task-for-the-pascal-voc-challenge). 
- In practice, all above this means that during train/test the displayed mAP will usually be quite a bit lower than reference found in published YOLO paper, as it is only calculated on one fixed confidence level. The final Pascal VOC metric takes into account best confidence scenario that maxes out precision.

## Results

In short, training only on Pascal VOC gave me about mAP\@0.5 = 0.2 (calculated with Padilla et al. scripts); when doing the Darknet pre-train on ImageNet I managed to obtain 0.58 (essentially same as original YOLO paper). With a fixed 0.4 confidence threshold those experiments correspond to 0.04 and 0.3 mAP.

## Notes for running the code
- first epoch always takes a bit longer to train
- all paths in train/test/detect scripts need to be updated to match your dataset directories. 
- this repository does not include the full datasets required to train the model, however I included tiny samples for both Pascal VOC and ImageNet to test things out more smoothly
- the in evaluation tables, I note the times and performances for multiple runs as I used a shared remote server that allowed running code only in certain blocks of time (e.g. 24h, 48h etc.)

### YOLO_v1 training (Object Detection task using different datasets)

**1. 8examples.csv** ([sample output](./sample_output/sample_output_original.txt))
- train.py/train_loader "drop_last" needs to be set to False in order not to skip the tiny 8 sample train set
- calculating one epoch on CPU even for such small data takes a bit so don't get dicouraged too fast (~100 epochs took ~30 min on Macbook Pro 2018)
- it took ~50 epochs to get loss <0.15, which is when mAP starts to be > 0.0
- it took ~80 epochs to get loss <0.0148, which gives mAP ~0.99 (the last few epochs it overfits very fast)

**2. 200examples.csv** ([sample output](./sample_output/sample_output_200examples.txt))
- the model training starts giving reasonable results faster than small training sample. At about 3 epochs mAP >0; around 15 epoch reaches 0.5 mAP, 25 epochs = 0.83 mAP, 100 epochs = 0.95 mAP. Train time for 100 epochs on single V100 is several minutes.

**3. train.csv (full Pascal VOD, 16k train, 5k test)** ([sample output](./sample_output/sample_output_fullVOC_100epochs.txt))
- on single V100 training 1st epoch takes about 26min with default settings (18m for train, 8m for accuracy calc)
- afterwards 1 epoch takes about 15min with default settings (8m for train, 7m for accuracy calc), 50 epochs about 10h, 100 epochs about 35h, 135 epochs as in paper takes xx hours
- performance on train (0.4 confidence threshold, 0.5 IoU): after 1st epoch, mAP = 0.004, after 2nd epoch, mAP = 0.005, after 10 epochs mAP = 0.054, after 50 epochs, mAP = 0.700, after 100 epochs mAP = 0.76, after 135 epochs mAP = 0.65
- performance on test (0.4 confidence threshold, 0.5 IoU): ;50 epochs, mAP = 0.045; 100 epochs, mAP = 0.040; ; 135 epochs,  mAP= 0.037 ; using test script on test8.csv = 0.86 mAP
- performance on test (0.4 confidence threshold, 0.5 IoU): 50 epoch / test8.csv mAP = 0.22 ; 100 epoch / test8.csv mAP=0.11 (seems to overfit heavily, mAP on train set is about 0.65), performance on test200, test1000 etc. doesnt change much vs. epochs
- performance on test (mAP at all confidence threshold, Pascal VOC metrics): trained only on Pascal VOC mAP@\0.5 = 0.2; pre-trained on ImageNet mAP\@0.5 = 0.58

### Darknet backbone training results (classification)

**SINGLE GPU (1x V100, batch size: 5 , lr: 2e-5 )**

|Run|Epoch start|Epoch end|Time (h)|Val Acc1 (%)|Val Acc5 (%)|Train Loss|
|---|-----------|---------|--------|------------|------------|----------|
|1  |0          |2        |17      |10.56       |27.114      |5.14      |
|2  |3          |5        |13      |19.164      |41.94       |4.43      |
|3  |6          |8 (10)   |24      |??          |??          |4         |
|4  |9          |11       |14      |31.9        |57.64       |3.5       |
|5  |12         |14       |17      |35.82       |61.65       |3.32      |
|6  |15         |18       |14      |38.41       |64.65       |2.75      |
|7  |19         |22       |15      |40.52       |66.55       |2.26      |
|8  |23         |26       |17      |42.13       |67.99       |2.03      |
|9  |27         |30       |17      |43.31       |69.15       |1.67      |
|10 |31         |33       |16      |44.4        |70.16       |1.45      |
|11 |34         |36       |16      |45.19       |70.79       |1.35      |
|12 |37         |41       |16      |45.98       |71.37       |1.45      |
|13 |42         |44       |17      |46.24       |71.83       |1.41      |
|14 |45         |47       |16      |46.478      |71.938      |1.3       |


**MULTI-GPU (4x V100 , batch size: 64 , lr: 2e-5 / batch size: 5 , lr: 2e-5 )**

|Run|Epoch start|Epoch end|Time (h)|Val Acc1 (%)|Val Acc5 (%)|Train Loss|comments                                                          |
|---|-----------|---------|--------|------------|------------|----------|------------------------------------------------------------------|
|1  |0          |9        |11      |1.99        |6.49        |6.55      |                                                                  |
|2  |10         |44       |43      |7.18        |20.01       |4.94      |                                                                  |
|3  |45         |77 (80)  |48+ (exceeded)|            |            |4.64      |Increased learning rate from 2e-5 -> 2e-4                         |
|4  |77         |80       |6       |13.11       |33.17       |6.03      |Same LR / batch size as single gpu ( bs =5 , lr = 2e-5) / 3epoch  |
|5  |81         |100      |37      |26.53       |50.66       |3.83      |Same LR / batch size as single gpu ( bs =5 , lr = 2e-5) / 20 epoch|
|6  |101        |120      |37h     |36.76       |61.69       |3.04      |( bs =5 , lr = 2e-5) / 20 epoch                                   |
|7  |121        |140      |37h     |43.07       |67.58       |2.67      |( bs =5 , lr = 2e-5) / 20 epoch                                   |
|8  |141        |160      |36h     |47.20       |71.29       |2.43      |( bs =5 , lr = 2e-5) / 20 epoch                                   |

* I also experimented with increased/decreasd learning rates but that didn't really give much improvement in training speed vs accuracy.

---

**Sources:**
- YOLO v1 paper by Joseph Redmon et al.: ["You Only Look Once: unified, Real-Time Object Detection"](https://arxiv.org/abs/1506.02640)
- YOLO v1 coding tutorial by Aladdin Persson: [Youtube](https://www.youtube.com/watch?v=n9_XyCGr-MI]) | [Git](https://github.com/aladdinpersson/Machine-Learning-Collection)
- Evaluation toolkit by Padilla et al.: ["A Comparative Analysis of Object Detection Metrics with a Companion Open-Source Toolkit"](https://github.com/rafaelpadilla/Object-Detection-Metrics)