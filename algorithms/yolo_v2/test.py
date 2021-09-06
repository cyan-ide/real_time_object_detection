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
	plot_image,
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
DATASET_DIR = "data_big/"
DATASET_DIR = "data_small/"
IMG_DIR = DATASET_DIR+"images/"
LABEL_DIR = DATASET_DIR+"labels/"
#TEST_ANNOTATIONS_FILE = DATASET_DIR+"100examples.csv" #"test.csv"
TEST_ANNOTATIONS_FILE = DATASET_DIR+"1examples.csv" #"test.csv"

#model location
LOAD_MODEL_FILE = "last.pth.tar"
#LOAD_MODEL_FILE = "overfit.pth.tar"

# Generic deep learning hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cpu" #DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16 # (64 in original paper)
WEIGHT_DECAY = 0
NUM_WORKERS = 0 # disable parallel
PIN_MEMORY = True

def main():
	# setup model
	darknet = Darknet19(classification_head = False).to(DEVICE) #.to(DEVICE) #create darknet without the classification part only used for pre-training on Imagenet
	#yolo v2
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

	#calculate mAP of the model
	pred_boxes, target_boxes = get_bboxes(test_loader, model, iou_threshold=0.5, threshold=0.4, split_size=SPLIT_SIZE,anchor_count = NUM_BOXES, num_classes = NUM_CLASSES, anchors=anchors, device=DEVICE)
	mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
	print("--Test mAP: {}".format(mean_avg_prec))

if __name__ == "__main__":
	main()