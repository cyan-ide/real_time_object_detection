# pytorch imports
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision.datasets import ImageFolder

import os

# components of yolo_v1 implementation
from darknet_model import Darknet

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
TEST_ANNOTATIONS_FILE = DATASET_DIR+"100examples.csv"

#train settings
LOAD_MODEL = True # load pre-trained weights ?
LOAD_MODEL_FILE = "darknet.e2.pth.tar"
WORK_DIR_PATH = "./work_dirs/"
WORK_DIR_NAME = "darknet_imagenet_test1/"
SAVE_MODEL_FILE = WORK_DIR_PATH+WORK_DIR_NAME+"darknet"
VALIDATE = True
VALIDATE_EPOCH_INTERVAL = 1 #
CHECKPOINT_EPOCH_INTERVAL = 2
CHECKPOINT_FINAL_EPOCH = True

# Generic deep learning hyperparameters
LEARNING_RATE = 2e-5 #0.1
MOMENTUM = 0.9
DEVICE = "cuda" #DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 5 # (64 in original paper)
WEIGHT_DECAY = 1e-4 #0 #1e-4
EPOCHS = 3
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

def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0) #target contains class numbers

		_, pred = output.topk(maxk, 1, True, True) #take indices of the top probabilities (ie. index = class number)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred)) #check which detected classes match target ground truth

		res = []
		for k in topk:
			correct_k = torch.reshape(correct[:k],(-1,)).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res

def main():
	# setup model
	model = Darknet(classification_head = True).to(DEVICE) #create model
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
	#optimizer = optim.Adam( model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY ) #create optimizer

	# load pre-trained weights if needed
	if LOAD_MODEL:
		load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

	# setup datasets
	transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),]) # image transformations

	# Data loading code
	#traindir = os.path.join('data_imagenet/small/', 'train/')
	traindir = "/custom_path_to_train_directory/train/"
	testdir = "/custom_path_to_test_directory/val/"

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])
	train_dataset = ImageFolder(
		traindir,
		transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]))
	test_dataset = ImageFolder(
		testdir,
		transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
			normalize,
		]))
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
		mean_acc1 = []
		mean_acc5 = []
		#validation stats
		mean_loss_val = []
		mean_acc1_val = []
		mean_acc5_val = []
		for batch_idx, (x, y) in enumerate(loop): #iterate batches
			x, y = x.to(DEVICE), y.to(DEVICE) 
			out = model(x)
			loss = loss_fn(out, y)

			acc1, acc5 = accuracy(out, y, topk=(1, 5))
			mean_acc1.append(acc1.item())
			mean_acc5.append(acc5.item())
			mean_loss.append(loss.item())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# update progress bar
			loop.set_postfix(loss=loss.item())

		print("--Mean loss: {} --Mean Acc1: {} % --Mean Acc5: {} %".format(sum(mean_loss)/len(mean_loss), sum(mean_acc1)/len(mean_acc1), sum(mean_acc5)/len(mean_acc5) ))

		#validate
		if (VALIDATE): 
			if ( epoch % VALIDATE_EPOCH_INTERVAL == 0):
				loop_val = tqdm(test_loader, leave=True) #show progress bar
				for batch_val_idx, (x_val, y_val) in enumerate(loop_val): #iterate batches
					x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE) 
					out_val = model(x_val)
					loss_val = loss_fn(out_val, y_val)
					acc1_val, acc5_val = accuracy(out_val, y_val, topk=(1, 5))
					mean_loss_val.append(loss_val.item())
					mean_acc1_val.append(acc1_val.item())
					mean_acc5_val.append(acc5_val.item())
				print("--(VAL) Mean loss: {} --Mean Acc1: {} % --Mean Acc5: {} %".format(sum(mean_loss_val)/len(mean_loss_val), sum(mean_acc1_val)/len(mean_acc1_val), sum(mean_acc5_val)/len(mean_acc5_val) ))		
	
		#save final model into file
		if ( epoch % CHECKPOINT_EPOCH_INTERVAL == 0) or (CHECKPOINT_FINAL_EPOCH == True and epoch == (EPOCHS-1)):
			checkpoint = {
				"state_dict": model.state_dict(),
				"optimizer": optimizer.state_dict(),
			}
			save_checkpoint(checkpoint, filename=SAVE_MODEL_FILE+".e"+str(epoch)+".pth.tar")

if __name__ == "__main__":
	main()
