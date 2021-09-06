# pytorch imports
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader

# components of yolo_v1 implementation
from model import YOLO_v2
from darknet_model import Darknet19
from dataset import YOLOv2_dataset, train_transforms, test_transforms
from loss import YOLOv2_loss
# utils taken from Aladdin Persson "from scrach" implementation series
from utils import (
	non_max_suppression,
	mean_average_precision,
	intersection_over_union,
	cellboxes_to_boxes,
	get_bboxes, 
	cells_to_bboxes, 
	plot_image,
	save_checkpoint,
	load_checkpoint,
)

from PIL import Image
import pandas as pd
import os
import numpy as np

seed = 123
torch.manual_seed(seed)

#YOLO_v2 model parameters
SPLIT_SIZE=13
NUM_BOXES=5
NUM_CLASSES=20
ANCHORS = torch.tensor( [(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)] )

#input data settings
DATASET_DIR = "data_voc/" 
IMG_INPUT_DIR = DATASET_DIR+"images/"
IMG_OUTPUT_DIR = DATASET_DIR+"images_out/"
DETECT_FILE_LIST = DATASET_DIR+"8examples.csv" 

#model location
LOAD_MODEL_FILE = "last.pth.tar"

#darknet tail
LOAD_DARKNET_MODEL = False #False
SAVE_DARKNET_MODEL_FILE = "darknet.last.pth.tar"

# Generic deep learning hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cpu" #DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 8 #16 # (64 in original paper)
WEIGHT_DECAY = 0
NUM_WORKERS = 0 # disable parallel
PIN_MEMORY = True

def main():
	# setup model
	darknet = Darknet19(classification_head = False).to(DEVICE) #.to(DEVICE) #create darknet without the classification part only used for pre-training on Imagenet
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
	#load_checkpoint(torch.load(LOAD_MODEL_FILE, map_location=torch.device('cpu') ), model, optimizer)
	load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

	# setup dataset
	filelist_csv = pd.read_csv(DETECT_FILE_LIST)
	transform = test_transforms

	for index in range(len(filelist_csv)):
		image_inpath = os.path.join(IMG_INPUT_DIR, filelist_csv.iloc[index, 0])
		image_outpath = os.path.join(IMG_OUTPUT_DIR, filelist_csv.iloc[index, 0])
		image = Image.open(image_inpath)
		# apply transformations is any (both image and bounding boxes)
		if transform:
			image = np.array(image.convert("RGB"))
			bounding_boxes = []
			augmentations = transform(image=image, bboxes=bounding_boxes)
			image = augmentations["image"]

		image = image.unsqueeze(0).to(DEVICE) #unsqueeze to (x,3,488,488) , where x is sample count (here will be 1)
		model.eval()
		out = model(image)
		out = out.reshape(-1, NUM_BOXES, NUM_CLASSES + 5, SPLIT_SIZE, SPLIT_SIZE)
		out = out.permute(0,1,3,4,2)

		bboxes = cells_to_bboxes(out, anchors, S=SPLIT_SIZE, is_preds=True, anchor_count = NUM_BOXES, num_classes = NUM_CLASSES)
		bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint") #remove excess bounding boxes

		plot_image(image[0].permute(1,2,0).to("cpu"), bboxes, output_filepath = image_outpath) #output image with boxes

if __name__ == "__main__":
	main()
