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

seed = 123
torch.manual_seed(seed)

#YOLO_v1 model parameters
SPLIT_SIZE=7
NUM_BOXES=2
NUM_CLASSES=20

#input data settings
DATASET_DIR = "data/"
IMG_DIR = DATASET_DIR+"images/"
LABEL_DIR = DATASET_DIR+"labels/"
TEST_ANNOTATIONS_FILE = DATASET_DIR+"100examples.csv" #"test.csv"

#model location
LOAD_MODEL_FILE = "overfit.pth.tar"

# Generic deep learning hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cpu" #DEVICE = "cuda" if torch.cuda.is_available else "cpu"
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
	model = YOLO_v1(split_size=SPLIT_SIZE, num_boxes=NUM_BOXES, num_classes=NUM_CLASSES).to(DEVICE) #create model
	optimizer = optim.Adam( model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY ) #create optimizer
	loss_fn = yolo_v1_loss() #creaate loss

	# load pre-trained weights if needed
	load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

	# setup dataset
	transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),]) # image transformations
	test_dataset = VOCDataset_custom(filelist_csv_path=TEST_ANNOTATIONS_FILE, transform=transform, image_dir=IMG_DIR, annotation_dir=LABEL_DIR)

	test_loader = DataLoader(
		dataset=test_dataset,
		batch_size=BATCH_SIZE,
		num_workers=NUM_WORKERS,
		pin_memory=PIN_MEMORY,
		shuffle=True,
		drop_last=True,
	)

	#calculate mAP of the model
	pred_boxes, target_boxes = get_bboxes(test_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE)
	mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
	print("--Test mAP: {}".format(mean_avg_prec))

if __name__ == "__main__":
	main()