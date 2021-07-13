# pytorch imports
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader

# components of yolo_v1 implementation
from model import YOLO_v1
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

from PIL import Image
import pandas as pd
import os

seed = 123
torch.manual_seed(seed)

#YOLO_v1 model parameters
SPLIT_SIZE=7
NUM_BOXES=2
NUM_CLASSES=20

#input data settings
DATASET_DIR = "data/" #"data_big/"
IMG_INPUT_DIR = DATASET_DIR+"images/"
IMG_OUTPUT_DIR = DATASET_DIR+"images_out/"
DETECT_FILE_LIST = DATASET_DIR+"8examples.csv"  #"100examples.csv" #"test.csv"

#model location
LOAD_MODEL_FILE = "last.pth.tar"

# Generic deep learning hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cpu" #DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 8 #16 # (64 in original paper)
WEIGHT_DECAY = 0
NUM_WORKERS = 0 # disable parallel
PIN_MEMORY = True

# image transformations (ie. augmentation)
class Compose(object):
	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, img):
		for t in self.transforms:
			img = t(img)

		return img

def main():

	# setup model
	model = YOLO_v1(split_size=SPLIT_SIZE, num_boxes=NUM_BOXES, num_classes=NUM_CLASSES).to(DEVICE) #create model
	optimizer = optim.Adam( model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY ) #create optimizer
	loss_fn = yolo_v1_loss() #creaate loss

	# load pre-trained weights if needed
	load_checkpoint(torch.load(LOAD_MODEL_FILE, map_location=torch.device('cpu') ), model, optimizer)

	# setup dataset
	filelist_csv = pd.read_csv(DETECT_FILE_LIST)
	transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),]) # image transformations

	for index in range(len(filelist_csv)):
		image_inpath = os.path.join(IMG_INPUT_DIR, filelist_csv.iloc[index, 0])
		image_outpath = os.path.join(IMG_OUTPUT_DIR, filelist_csv.iloc[index, 0])
		image = Image.open(image_inpath)
		# apply transformations is any (both image and bounding boxes)
		if transform:
			image = transform(image)
		image = image.unsqueeze(0).to(DEVICE) #unsqueeze to (x,3,488,488) , where x is sample count (here will be 1)
		model.eval()
		bboxes = cellboxes_to_boxes(model(image))
		bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint") #remove excess bounding boxes
		plot_image(image[0].permute(1,2,0).to("cpu"), bboxes, output_filepath = image_outpath) #output image with boxes

if __name__ == "__main__":
	main()
