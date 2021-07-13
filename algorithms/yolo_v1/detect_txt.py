# pytorch imports
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader

import os

# components of yolo_v1 implementation
from model import YOLO_v1
from darknet_model import Darknet
from dataset import VOCDataset_custom
from loss import yolo_v1_loss
# utils taken from Aladdin Persson "from scrach" implementation series
from utils import (
	non_max_suppression,
	mean_average_precision,
	intersection_over_union,
	cellboxes_to_boxes,
	get_bboxes,
	plot_image,
	save_checkpoint,
	load_checkpoint,
)

seed = 123
torch.manual_seed(seed)

#YOLO_v1 model parameters
SPLIT_SIZE=7
NUM_BOXES=2
NUM_CLASSES=20

#input data settings
DATASET_DIR = "data/" #"data_small/" #"data_big/"
IMG_DIR = DATASET_DIR+"images/"
LABEL_DIR = DATASET_DIR+"labels/"
TRAIN_ANNOTATIONS_FILE = DATASET_DIR+ "train.csv"
TEST_ANNOTATIONS_FILE = DATASET_DIR+ "test.csv" #"8examples.csv" #"100examples.csv" #"test.csv"

#model location
LOAD_MODEL_FILE =   "yolo_v1.100epochs.voc.pth.tar" 
#LOAD_MODEL_FILE = "yolov1_nopretrain/yolo_v1.50epochs.voc.pth.tar"
LOAD_MODEL_FILE = "last.pth.tar" 

#darknet tail
LOAD_DARKNET_MODEL = False #True #True #False
SAVE_DARKNET_MODEL_FILE = "darknet.e47.pth.tar"

#output directory
#full
OUTPUT_DIR = "data_annotations_output/" #the annotations for prediction and ground truth will be written into this directory
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
	darknet = Darknet(classification_head = False).to(DEVICE)
	if (LOAD_DARKNET_MODEL): #load pre-trained darknet weights (on imagenet)
		print("Loading darknet weights...")
		src_state_dict = torch.load(SAVE_DARKNET_MODEL_FILE)['state_dict']
		dst_state_dict = darknet.state_dict()
		#iterate over darknet and load weights (this is not done directly from src because it also has classification head weights)
		for k in dst_state_dict.keys():
			#print('Loading weight of', k)
			dst_state_dict[k] = src_state_dict[k]
		darknet.load_state_dict(dst_state_dict)
		print('Loaded {} darknet layer weights'.format(len(dst_state_dict.keys())))

	model = YOLO_v1(darknet_layers = darknet.darknet_layers, split_size=SPLIT_SIZE, num_boxes=NUM_BOXES, num_classes=NUM_CLASSES).to(DEVICE) #create model
	optimizer = optim.Adam( model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY ) #create optimizer
	loss_fn = yolo_v1_loss() #creaate loss

	# load pre-trained weights if needed
	load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

	# setup dataset
	transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),]) # image transformations
	# train_dataset = VOCDataset_custom(filelist_csv_path=TRAIN_ANNOTATIONS_FILE, transform=transform, image_dir=IMG_DIR, annotation_dir=LABEL_DIR)
	test_dataset = VOCDataset_custom(filelist_csv_path=TEST_ANNOTATIONS_FILE, transform=transform, image_dir=IMG_DIR, annotation_dir=LABEL_DIR)

	# train_loader = DataLoader(
	# 	dataset=train_dataset,
	# 	batch_size=BATCH_SIZE,
	# 	num_workers=NUM_WORKERS,
	# 	pin_memory=PIN_MEMORY,
	# 	shuffle=True,
	# 	drop_last=False,
	# )

	test_loader = DataLoader(
		dataset=test_dataset,
		batch_size=BATCH_SIZE,
		num_workers=NUM_WORKERS,
		pin_memory=PIN_MEMORY,
		#shuffle=True,
		shuffle=False,
		drop_last=False,
	)

	#calculate mAP of the model
	pred_boxes, target_boxes = get_bboxes(test_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE)
	mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
	print("--Test mAP: {}".format(mean_avg_prec))

	#test v2
	#pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE)
	#mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
	#print("--Train mAP: {}".format(mean_avg_prec))
	#exit()
	batch_idx = 0

	#check if output dir exists
	if os.path.isdir(OUTPUT_DIR) == False:
		print("Creating output directories...")
		os.mkdir(OUTPUT_DIR)
		os.mkdir(OUTPUT_GROUND_TRUTH_ANN)
		os.mkdir(OUTPUT_PREDICTION_ANN)

	for x, y in test_loader:
		x = x.to(DEVICE)
		#y = y.to(DEVICE)
		bboxes = cellboxes_to_boxes(model(x))
		true_bboxes = cellboxes_to_boxes(y)
		#print("true_bboxes_all - len: {}, output_size: {}".format(len(true_bboxes[0]), len(y[0]) ))
		for idx in range(len(bboxes)): #for each image in batch
			#print("idx: {}".format(idx))
			#print(test_dataset.filelist_csv.iloc[BATCH_SIZE*batch_idx+idx, 0])
			#bboxes_img = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
			bboxes_img = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.0, box_format="midpoint")
			#plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes_img, output_filepath ="data_small/test_old/images_out/"+test_dataset.filelist_csv.iloc[BATCH_SIZE*batch_idx+idx, 0]) #str(idx)+".png")
			#print("---- PRED: -----")
			with open(OUTPUT_PREDICTION_ANN+test_dataset.filelist_csv.iloc[BATCH_SIZE*batch_idx+idx, 1], 'w') as f:
				for box in bboxes_img: #range(len(true_bboxes_img)):
					if box[1] > 0.0: #if prob greater than one output annotation
						f.write("{} {} {} {} {} {}\n".format(box[0], box[1], box[2], box[3], box[4], box[5]))
						#f.writelines(lines)
						#print("{} {} {} {} {} {}".format(box[0], box[1], box[2], box[3], box[4], box[5]))


			true_bboxes_img = true_bboxes[idx]
			#plot_image(x[idx].permute(1,2,0).to("cpu"), true_bboxes_img, output_filepath ="data_small/test_old/images_out_groundt/"+test_dataset.filelist_csv.iloc[BATCH_SIZE*batch_idx+idx, 0]) #+str(idx)+".png")
			#print("---- G TRUTH -----")
			with open(OUTPUT_GROUND_TRUTH_ANN+test_dataset.filelist_csv.iloc[BATCH_SIZE*batch_idx+idx, 1], 'w') as f:
			#with open("data_small/test_old/images_out_fake/"+test_dataset.filelist_csv.iloc[BATCH_SIZE*batch_idx+idx, 1], 'w') as f:
				for box in true_bboxes_img: #range(len(true_bboxes_img)):
					if box[1] > 0.0: #if prob greater than one output annotation
						f.write("{} {} {} {} {}\n".format(box[0], box[2], box[3], box[4], box[5]))
						#f.write("{} {} {} {} {} {}\n".format(box[0], box[1], box[2], box[3], box[4], box[5]) )
						#print("{} {} {} {} {}".format(box[0], box[1], box[2], box[3], box[4], box[5]))
		batch_idx += 1

if __name__ == "__main__":
	main()
