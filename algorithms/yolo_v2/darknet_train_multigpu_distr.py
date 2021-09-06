# pytorch imports
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader
#from torchvision.datasets import ImageNet
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import MultiplicativeLR

import os

# components of yolo_v1 implementation
from darknet_model import Darknet19

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

#disrubuted settings
WORLD_SIZE = 4 #gpu count

#YOLO_v1 model parameters
SPLIT_SIZE=7
NUM_BOXES=2
NUM_CLASSES=20

#input data settings
DATASET_DIR = "data_big/"
IMG_DIR = DATASET_DIR+"images/"
LABEL_DIR = DATASET_DIR+"labels/"
TRAIN_ANNOTATIONS_FILE = DATASET_DIR+"100examples.csv"
TEST_ANNOTATIONS_FILE = DATASET_DIR+"100examples.csv" #"test.csv"

#train settings
LOAD_MODEL = True # load pre-trained weights ?
LOAD_MODEL_FILE = "./darknet.previous.pth.tar"
WORK_DIR_PATH = "./work_dirs/"
WORK_DIR_NAME = "darknet19_imagenet_full_dist_latest/" 

SAVE_MODEL_FILE = WORK_DIR_PATH+WORK_DIR_NAME+"darknet"
VALIDATE = True
VALIDATE_EPOCH_INTERVAL = 10 
VALIDATE_FINAL_EPOCH = True
CHECKPOINT_EPOCH_INTERVAL = 100
CHECKPOINT_FINAL_EPOCH = True


RUN_EPOCHS = 30 #30 #how many epoch to run in this script
STARTING_EPOCH_NO = 120 #29 #starting epoch no for this script

# Generic deep learning hyperparameters
ADJUST_DECAYING_LEARNING_RATE = True
LEARNING_RATE = 1 #2e-4 #0.1
LEARNING_RATE_DECAY_EXPONENT = 0.1 #4
MOMENTUM = 0.9
DEVICE = "cuda" #DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 64 #64 #5 #64 #6 # <-- per worker #192  # (64 in original paper)
WEIGHT_DECAY = 1e-4 #0 #1e-4
NUM_WORKERS = 0 # disable parallel
PIN_MEMORY = True
TOTAL_EPOCHS = 160 #total epoch count for experiment (this is needed for calculation of learning rate decay, which gradually goes to zero by the end of learning)
INPUT_IMG_SIZE = 224

#finetuneing (final epochs are trained with higher-resolution)
FINE_TUNE = True
FINE_TUNE_TOTAL_EPOCHS = 10
FINE_TUNE_RUN_EPOCHS = 10
FINE_TUNE_START_EPOCH = 0
FINE_TUNE_LEARNING_RATE= 0.001
FINE_TUNE_INPUT_IMG_SIZE = 448
FINE_TUNE_BATCH_SIZE = 16
FINE_TUNE_ADJUST_DECAYING_LEARNING_RATE = False
FINE_TINE_SAVE_MODEL_FILE = WORK_DIR_PATH+WORK_DIR_NAME+"darknet19_150e"+".fine_tune"

if FINE_TUNE: #Yolo_v2 has a final fine-tune stage where its trained for several epochs on higher resolution images
	TOTAL_EPOCHS = FINE_TUNE_TOTAL_EPOCHS
	LEARNING_RATE = FINE_TUNE_LEARNING_RATE
	INPUT_IMG_SIZE = FINE_TUNE_INPUT_IMG_SIZE
	BATCH_SIZE = FINE_TUNE_BATCH_SIZE
	#
	ADJUST_DECAYING_LEARNING_RATE = FINE_TUNE_ADJUST_DECAYING_LEARNING_RATE
	RUN_EPOCHS = FINE_TUNE_RUN_EPOCHS
	STARTING_EPOCH_NO = FINE_TUNE_START_EPOCH
	#
	SAVE_MODEL_FILE = FINE_TINE_SAVE_MODEL_FILE


#adjust initial learning rate given starting epoch number (assuming LEARNING_RATE is at epoch 0)
if ADJUST_DECAYING_LEARNING_RATE and STARTING_EPOCH_NO !=0:
	for epoch_no in range(STARTING_EPOCH_NO+1):
		LEARNING_RATE = LEARNING_RATE * ( (1 - epoch_no / TOTAL_EPOCHS) ** LEARNING_RATE_DECAY_EXPONENT )
		#print("EPOCH_NO: {} -- LR: {} (calc: {} -  ( (1 - {} / {}) ** {} ) )".format(epoch_no,LEARNING_RATE,LEARNING_RATE,epoch_no,TOTAL_EPOCHS,LEARNING_RATE_DECAY_EXPONENT))
	print("ADJUSTED STARTING LEARNING RATE (for start epoch {}): {}".format(STARTING_EPOCH_NO,LEARNING_RATE))

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

def main_worker(DEVICE_RANK):
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '12355'
	dist_url = "tcp://localhost:12355"
	dist_backend = "nccl"
	dist.init_process_group(backend=dist_backend, init_method=dist_url, world_size=WORLD_SIZE, rank=DEVICE_RANK)
	# setup model
	model = Darknet19(classification_head = True).to(DEVICE_RANK) #create model
	model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[DEVICE_RANK])

	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
	if (ADJUST_DECAYING_LEARNING_RATE):
		lmbda = lambda epoch: (1 - (STARTING_EPOCH_NO+epoch) / (TOTAL_EPOCHS)) ** LEARNING_RATE_DECAY_EXPONENT #0.95
	else:
		lmbda = lambda epoch: 1
	scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)

	# load pre-trained weights if needed
	if LOAD_MODEL:
		map_location = {'cuda:%d' % 0: 'cuda:%d' % DEVICE_RANK}
		load_checkpoint(torch.load(LOAD_MODEL_FILE, map_location=map_location), model, optimizer)

	# setup datasets
	transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),]) # image transformations

	# Data loading code
	# (custom paths)
	traindir = "./imagenet/jpeg/train/"
	testdir = "./imagenet_data/val/"

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])
	train_dataset = ImageFolder(
		traindir,
		transforms.Compose([
			transforms.RandomResizedCrop(INPUT_IMG_SIZE),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]))
	test_dataset = ImageFolder(
		testdir,
		transforms.Compose([
            transforms.Resize(INPUT_IMG_SIZE+32),
            transforms.CenterCrop(INPUT_IMG_SIZE),
            transforms.ToTensor(),
			normalize,
		]))

	train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
	train_loader = DataLoader(
		dataset=train_dataset,
		batch_size=BATCH_SIZE,
		num_workers=NUM_WORKERS,
		pin_memory=PIN_MEMORY,
		shuffle=False,
		drop_last=False,
		sampler=train_sampler,
	)

	test_loader = DataLoader(
		dataset=test_dataset,
		batch_size=BATCH_SIZE,
		num_workers=NUM_WORKERS,
		pin_memory=PIN_MEMORY,
		shuffle=True,
		drop_last=False,
	)

	#pre-train validation if loading pre-trained model (to verify accuracy before training futher)
	if (LOAD_MODEL and VALIDATE): 
		#if DEVICE_RANK == 0:
		model.eval()
		if DEVICE_RANK == 0:
			loop_val = tqdm(test_loader, leave=True, desc= "[VAL-PRE-TRAIN]") #show progress bar (only for master process)
		else:
			loop_val = tqdm(test_loader, leave=True, desc= "[VAL-PRE-TRAIN]", disable = True) #test_loader (disable progress from slaves)

		#validation stats
		mean_loss_val = []
		mean_acc1_val = []
		mean_acc5_val = []
		for batch_val_idx, (x_val, y_val) in enumerate(loop_val): #iterate batches
			x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE_RANK) 
			out_val = model(x_val)
			loss_val = loss_fn(out_val, y_val)
			acc1_val, acc5_val = accuracy(out_val, y_val, topk=(1, 5))
			mean_loss_val.append(loss_val.item())
			mean_acc1_val.append(acc1_val.item())
			mean_acc5_val.append(acc5_val.item())
		if DEVICE_RANK == 0:
			print("--(VAL-PRE-TRAIN) Mean loss: {} --Mean Acc1: {} % --Mean Acc5: {} %".format(sum(mean_loss_val)/len(mean_loss_val), sum(mean_acc1_val)/len(mean_acc1_val), sum(mean_acc5_val)/len(mean_acc5_val) ))

	# run training
	lrs = []
	for epoch in range(STARTING_EPOCH_NO, STARTING_EPOCH_NO+RUN_EPOCHS):
		if DEVICE_RANK == 0:
			print("Epoch {}/{} (LEARNING RATE: {})".format(epoch,TOTAL_EPOCHS-1,optimizer.param_groups[0]["lr"]))
		train_sampler.set_epoch(epoch)

		if DEVICE_RANK == 0:
			loop = tqdm(train_loader, leave=True, desc="[TRAIN]") #show progress bar
		else:
			loop = tqdm(train_loader, leave=True, desc="[TRAIN]", disable = True) #for non-main processes dont show progress bar
		mean_loss = []
		mean_acc1 = []
		mean_acc5 = []
		#validation stats
		mean_loss_val = []
		mean_acc1_val = []
		mean_acc5_val = []
		model.train()
		for batch_idx, (x, y) in enumerate(loop): #iterate batches
			x, y = x.to(DEVICE_RANK), y.to(DEVICE_RANK) 
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

		if DEVICE_RANK == 0:
			print("--Mean loss: {} --Mean Acc1: {} % --Mean Acc5: {} %".format(sum(mean_loss)/len(mean_loss), sum(mean_acc1)/len(mean_acc1), sum(mean_acc5)/len(mean_acc5) ))

		#validate
		if (VALIDATE): 
			if ( epoch % VALIDATE_EPOCH_INTERVAL == 0) or (VALIDATE_FINAL_EPOCH == True and epoch == ((STARTING_EPOCH_NO+RUN_EPOCHS)-1)):
				#if DEVICE_RANK == 0:
				model.eval()
				if DEVICE_RANK == 0:
					loop_val = tqdm(test_loader, leave=True, desc= "[VAL]") #show progress bar (only for master process)
				else:
					loop_val = tqdm(test_loader, leave=True, desc= "[VAL]", disable = True) #test_loader (disable progress from slaves)
				for batch_val_idx, (x_val, y_val) in enumerate(loop_val): #iterate batches
					x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE_RANK) 
					out_val = model(x_val)
					loss_val = loss_fn(out_val, y_val)
					acc1_val, acc5_val = accuracy(out_val, y_val, topk=(1, 5))
					mean_loss_val.append(loss_val.item())
					mean_acc1_val.append(acc1_val.item())
					mean_acc5_val.append(acc5_val.item())
				if DEVICE_RANK == 0:
					print("--(VAL) Mean loss: {} --Mean Acc1: {} % --Mean Acc5: {} %".format(sum(mean_loss_val)/len(mean_loss_val), sum(mean_acc1_val)/len(mean_acc1_val), sum(mean_acc5_val)/len(mean_acc5_val) ))		
	
		#save final model into file
		if DEVICE_RANK == 0: #save only in first worker (all others are synced so doesnt matter)
			if (epoch != 0) and ( (epoch % CHECKPOINT_EPOCH_INTERVAL == 0) or (CHECKPOINT_FINAL_EPOCH == True and epoch == (STARTING_EPOCH_NO+RUN_EPOCHS-1)) ):
				checkpoint = {
					"state_dict": model.state_dict(),
					"optimizer": optimizer.state_dict(),
				}
				save_checkpoint(checkpoint, filename=SAVE_MODEL_FILE+".e"+str(epoch)+".pth.tar")

		lrs.append(optimizer.param_groups[0]["lr"])
		scheduler.step()
	if DEVICE_RANK == 0:
		print(lrs)
	dist.destroy_process_group()

def main():
	mp.spawn(main_worker, nprocs=WORLD_SIZE)

if __name__ == "__main__":
	main()
