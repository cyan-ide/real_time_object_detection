# use this script to export YOLO_V2 predictions into txt file (later those can be used with external evaluation libraries with all sorts of metrics)

# pytorch imports
import torch
import torchvision.transforms as transforms
import torchvision.ops

import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader

import os

# components of yolo_v1 implementation
from model import YOLO_v2
from darknet_model import Darknet19
from dataset import YOLOv2_dataset, train_transforms, test_transforms
from loss import YOLOv2_loss
# utils taken from Aladdin Persson "from scrach" implementation series
from utils import (
	mean_average_precision,
	get_bboxes,
	cells_to_bboxes, 
	xywhn2xyxy,
	save_checkpoint,
	load_checkpoint,
)

seed = 123
torch.manual_seed(seed)

#YOLO_v2 model parameters
SPLIT_SIZE=13
NUM_BOXES=5
NUM_CLASSES=20
ANCHORS = torch.tensor( [(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)] )

#input data settings
DATASET_DIR = "data_voc/"
IMG_DIR = DATASET_DIR+"images/"
LABEL_DIR = DATASET_DIR+"labels/"
TRAIN_ANNOTATIONS_FILE = DATASET_DIR+ "8examples.csv"
TEST_ANNOTATIONS_FILE = DATASET_DIR+ "8examples.csv" 

#model location
LOAD_MODEL_FILE =   "last.pth.tar" 

#darknet tail
LOAD_DARKNET_MODEL = False 
SAVE_DARKNET_MODEL_FILE = "./darknet.last.pth.tar"

#output directory
OUTPUT_DIR = "./results/eval_data/run1/"
OUTPUT_GROUND_TRUTH_ANN = OUTPUT_DIR+"groundt/"
OUTPUT_PREDICTION_ANN = OUTPUT_DIR+"predictions/"

# Generic deep learning hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cuda" #DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16 # (64 in original paper)
WEIGHT_DECAY = 0
NUM_WORKERS = 0 # disable parallel
PIN_MEMORY = True

# image transformations (ie. augmentation)
class Compose(object):
	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, img, bboxes):
		for t in self.transforms:
			img, bboxes = t(img), bboxes

		return img, bboxes

def main():
	# setup model
	darknet = Darknet19(classification_head = False).to(DEVICE) #create darknet without the classification part only used for pre-training on Imagenet
	if (LOAD_DARKNET_MODEL): #load pre-trained darknet weights (on imagenet)
		print("Loading darknet weights...")
		src_state_dict = torch.load(SAVE_DARKNET_MODEL_FILE)['state_dict']
		dst_state_dict = darknet.state_dict()
		#iterate over darknet and load weights (this is not done directly from src because it also has classification head weights)
		for k in dst_state_dict.keys(): 
			dst_state_dict[k] = src_state_dict[k]
		darknet.load_state_dict(dst_state_dict)
		print('Loaded {} darknet layer weights'.format(len(dst_state_dict.keys())))

	anchors = ANCHORS.to(DEVICE)
	model = YOLO_v2(darknet_layers = darknet.darknet_layers, split_size=SPLIT_SIZE, num_boxes=NUM_BOXES, num_classes=NUM_CLASSES).to(DEVICE) #create Yolo model with darknet in tail
	optimizer = optim.Adam( model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY ) #create optimizer
	loss_fn = YOLOv2_loss(anchors= anchors) #creaate loss

	# load pre-trained weights if needed
	load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

	# setup dataset
	transform = test_transforms
	test_dataset = YOLOv2_dataset(filelist_csv_path=TEST_ANNOTATIONS_FILE, transform=test_transforms, image_dir=IMG_DIR, annotation_dir=LABEL_DIR, split_size=SPLIT_SIZE, num_boxes=NUM_BOXES, num_classes=NUM_CLASSES, anchors = ANCHORS)
	test_loader = DataLoader(
		dataset=test_dataset,
		batch_size=BATCH_SIZE,
		num_workers=NUM_WORKERS,
		pin_memory=PIN_MEMORY,
		shuffle=False,
		drop_last=False,
	)

	#check if output dir exists
	if os.path.isdir(OUTPUT_DIR) == False:
		print("Creating output directories...")
		os.mkdir(OUTPUT_DIR)
		os.mkdir(OUTPUT_GROUND_TRUTH_ANN)
		os.mkdir(OUTPUT_PREDICTION_ANN)

	#calculate mAP of the model
	pred_boxes, target_boxes = get_bboxes(test_loader, model, iou_threshold=0.5, threshold=0.4, split_size=SPLIT_SIZE,anchor_count = NUM_BOXES, num_classes = NUM_CLASSES, anchors=anchors, device=DEVICE)
	mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
	print("--Test mAP: {}".format(mean_avg_prec))

	batch_idx = 0
	nms_threshold = 0.0
	iou_threshold = 0.5
	model.eval()
	for x, y in test_loader:
		x = x.to(DEVICE)

		out = model(x)
		out = out.reshape(-1, NUM_BOXES, NUM_CLASSES + 5, SPLIT_SIZE, SPLIT_SIZE)
		out = out.permute(0,1,3,4,2)

		bboxes = cells_to_bboxes(out, anchors, S=SPLIT_SIZE, is_preds=True, anchor_count = NUM_BOXES, num_classes = NUM_CLASSES)
		true_bboxes = cells_to_bboxes(y, anchors, S=SPLIT_SIZE, is_preds=False, anchor_count = NUM_BOXES, num_classes = NUM_CLASSES)

		for idx in range(len(bboxes)): #for each image in batch

			#non-max surpression for predictions for given sample
			bboxes_subset = [box for box in bboxes[idx] if box[1] > nms_threshold] #filter out only above given confidence threshold
			if len(bboxes_subset) > 0:
				bboxes_subset_scores = torch.tensor(bboxes_subset)[...,1]
				bboxes_subset = torch.tensor(bboxes_subset)[...,2:]
				bboxes_subset_converted = xywhn2xyxy(bboxes_subset, w=x.shape[2], h=x.shape[3])
				nms_indices = torchvision.ops.nms(boxes= bboxes_subset_converted, scores= bboxes_subset_scores, iou_threshold= iou_threshold)
				nms_boxes = [bboxes[idx][i] for i in nms_indices]
			else:
				nms_boxes = []

			#save predictions
			with open(OUTPUT_PREDICTION_ANN+test_dataset.filelist_csv.iloc[BATCH_SIZE*batch_idx+idx, 1], 'w') as f:
				for box in nms_boxes: 
					if box[1] > 0.0: #if prob greater than zero, output annotation
						f.write("{} {} {} {} {} {}\n".format(box[0], box[1], box[2], box[3], box[4], box[5]))

			#save ground truth
			with open(OUTPUT_GROUND_TRUTH_ANN+test_dataset.filelist_csv.iloc[BATCH_SIZE*batch_idx+idx, 1], 'w') as f:
				for box in true_bboxes[idx]: 
					if box[1] > 0.0: #if prob greater than zero, output annotation
						f.write("{} {} {} {} {}\n".format(box[0], box[2], box[3], box[4], box[5]))
		batch_idx += 1

if __name__ == "__main__":
	main()
