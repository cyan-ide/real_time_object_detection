# pytorch imports
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader

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
DATASET_DIR = "data/"
IMG_DIR = DATASET_DIR+"images/"
LABEL_DIR = DATASET_DIR+"labels/"
TRAIN_ANNOTATIONS_FILE = DATASET_DIR+"100examples.csv"
TEST_ANNOTATIONS_FILE = DATASET_DIR+"100examples.csv" #"test.csv"

#train settings
LOAD_MODEL = False # load pre-trained weights ?
#LOAD_MODEL_FILE = "overfit.pth.tar"
SAVE_MODEL_FILE = "last.pth.tar"

#darknet tail
LOAD_DARKNET_MODEL = True #False
SAVE_DARKNET_MODEL_FILE = "darknet.last.pth.tar"

# Generic deep learning hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cpu" #DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16 # (64 in original paper)
WEIGHT_DECAY = 0
EPOCHS = 5
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
	darknet = Darknet(classification_head = False).to(DEVICE) #create darknet without the classification part only used for pre-training on Imagenet
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

	model = YOLO_v1(darknet_layers = darknet.darknet_layers, split_size=SPLIT_SIZE, num_boxes=NUM_BOXES, num_classes=NUM_CLASSES).to(DEVICE) #create Yolo model with darknet in tail
	optimizer = optim.Adam( model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY ) #create optimizer
	loss_fn = yolo_v1_loss() #creaate loss

	# load pre-trained weights if needed
	if LOAD_MODEL:
		load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

	# setup datasets
	transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),]) # image transformations
	train_dataset = VOCDataset_custom(filelist_csv_path=TRAIN_ANNOTATIONS_FILE, transform=transform, image_dir=IMG_DIR, annotation_dir=LABEL_DIR)
	test_dataset = VOCDataset_custom(filelist_csv_path=TEST_ANNOTATIONS_FILE, transform=transform, image_dir=IMG_DIR, annotation_dir=LABEL_DIR)

	train_loader = DataLoader(
		dataset=train_dataset,
		batch_size=BATCH_SIZE,
		num_workers=NUM_WORKERS,
		pin_memory=PIN_MEMORY,
		shuffle=True,
		drop_last=False,
	)

	test_loader = DataLoader(
		dataset=test_dataset,
		batch_size=BATCH_SIZE,
		num_workers=NUM_WORKERS,
		pin_memory=PIN_MEMORY,
		shuffle=True,
		drop_last=True,
	)

	# run training
	for epoch in range(EPOCHS):
		print("Epoch {}/{}".format(epoch,EPOCHS))
		loop = tqdm(train_loader, leave=True) #show progress bar
		mean_loss = []
		for batch_idx, (x, y) in enumerate(loop): #iterate batches
			x, y = x.to(DEVICE), y.to(DEVICE) 
			out = model(x)
			loss = loss_fn(out, y)
			mean_loss.append(loss.item())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# update progress bar
			loop.set_postfix(loss=loss.item())

		print("--Mean loss: {}".format(sum(mean_loss)/len(mean_loss)))

		#calculate mAP of the model
		pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE)
		mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
		print("--Train mAP: {}".format(mean_avg_prec))
	
	#save final model into file
	checkpoint = {
		"state_dict": model.state_dict(),
		"optimizer": optimizer.state_dict(),
	}
	save_checkpoint(checkpoint, filename=SAVE_MODEL_FILE)

if __name__ == "__main__":
	main()