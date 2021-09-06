# pytorch imports
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import time
import datetime



# components of yolo_v2 implementation
from model import YOLO_v2
from darknet_model import Darknet19
from dataset import YOLOv2_dataset, train_transforms, train_transforms2, test_transforms
from loss import YOLOv2_loss
# utils taken from Aladdin Persson "from scrach" implementation series
from utils import (
	mean_average_precision,
	get_bboxes, 
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

#train settings
LOAD_MODEL = False # load pre-trained weights ?
#LOAD_MODEL_FILE = "overfit.pth.tar"

SAVE_MODEL = True
SAVE_MODEL_FILE = "last.pth.tar"

WORK_DIR_PATH = "./work_dirs/"
WORK_DIR_NAME = "yolov2_train_16082021/"
LOG_EXTRA_NAME = "full_voc_lr_set2_"


#darknet tail
LOAD_DARKNET_MODEL = True #False
SAVE_DARKNET_MODEL_FILE = "./darknet.last.pth.tar"

EVAL_TRAIN = True #evaluate acc on train set during training
EVAL_EPOCH_INTERVAL = 10 #

# Generic deep learning hyperparameters
LEARNING_RATE = 2e-5 #1e-3 # #1e-4 #2e-5 #1e-3 #1e-3
LEARNING_RATE_SCHEDULE_ON = True
LEARNING_RATE_SCHEDULE = {"0": 1e-5, "5": 1e-4, "80": 1e-5, "110": 1e-6} #open source schedule
# LEARNING_RATE_SCHEDULE = {"0": 1e-3, "60": 1e-4, "80": 1e-5, "90": 1e-5} # yolo paper lr schedule

DEVICE = "cuda" #"cpu" #DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 10 #16 # (64 in original paper)
WEIGHT_DECAY = 0.0005 #0
MOMENTUM = 0.09 #0
EPOCHS = 160
# EPOCHS = 10
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
			dst_state_dict[k] = src_state_dict['module.'+k] #temp added "module." due to some changes in darknet training
		darknet.load_state_dict(dst_state_dict)
		print('Loaded {} darknet layer weights'.format(len(dst_state_dict.keys())))


	anchors = ANCHORS.to(DEVICE)
	model = YOLO_v2(darknet_layers = darknet.darknet_layers, split_size=SPLIT_SIZE, num_boxes=NUM_BOXES, num_classes=NUM_CLASSES).to(DEVICE) #create Yolo model with darknet in tail
	optimizer = optim.Adam( model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY ) #create optimizer
	#optimizer = optim.SGD( model.parameters(), lr=LEARNING_RATE, momentum= MOMENTUM, weight_decay= WEIGHT_DECAY)
	loss_fn = YOLOv2_loss(anchors= anchors) #creaate loss
	

	# load pre-trained weights if needed
	if LOAD_MODEL:
		load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

	# setup datasets
	train_dataset = YOLOv2_dataset(filelist_csv_path=TRAIN_ANNOTATIONS_FILE, transform=train_transforms, image_dir=IMG_DIR, annotation_dir=LABEL_DIR, split_size=SPLIT_SIZE, num_boxes=NUM_BOXES, num_classes=NUM_CLASSES, anchors = ANCHORS)
	test_dataset = YOLOv2_dataset(filelist_csv_path=TEST_ANNOTATIONS_FILE, transform=test_transforms, image_dir=IMG_DIR, annotation_dir=LABEL_DIR, split_size=SPLIT_SIZE, num_boxes=NUM_BOXES, num_classes=NUM_CLASSES, anchors = ANCHORS)

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
		shuffle=False,
		drop_last=False,
	)

	#logging
	log_dir = WORK_DIR_PATH+"logs/run_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	writer = SummaryWriter(log_dir,flush_secs=1) #'runs/test_exp1'

	# run training
	start_total_time = time.time()
	for epoch in range(EPOCHS):
		if LEARNING_RATE_SCHEDULE_ON and str(epoch) in LEARNING_RATE_SCHEDULE.keys():
			for param_group in optimizer.param_groups:
				param_group['lr'] = LEARNING_RATE_SCHEDULE[str(epoch)]
		# start = time.time()
		print("Epoch {}/{} (LEARNING RATE: {})".format(epoch,EPOCHS,optimizer.param_groups[0]["lr"]))
		loop = tqdm(train_loader, leave=True) #show progress bar
		#stats logging
		mean_loss = []
		mean_box_loss = []
		mean_object_loss = []
		mean_no_object_loss = []
		mean_class_loss = []

		start_time = time.time()
		for batch_idx, (x, y) in enumerate(loop): #iterate batches
			x, y = x.to(DEVICE), y.to(DEVICE)
			out = model(x)
			loss, box_loss, object_loss, no_object_loss, class_loss = loss_fn(out, y)

			mean_loss.append(loss.item())
			mean_box_loss.append(box_loss.item())
			mean_object_loss.append(object_loss.item())
			mean_no_object_loss.append(no_object_loss.item())
			mean_class_loss.append(class_loss.item())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# update progress bar
			loop.set_postfix(loss=loss.item())

		end_loss_time = time.time()
		end_loss_diff = end_loss_time - start_time

		#calculate mAP of the model if eval is on
		if EVAL_TRAIN and (epoch != 0) and (epoch % EVAL_EPOCH_INTERVAL == 0):
			pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4, split_size=SPLIT_SIZE,anchor_count = NUM_BOXES, num_classes = NUM_CLASSES, anchors=anchors, device=DEVICE)
			mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")

			end_epoch_time = time.time()
			end_epoch_diff = end_epoch_time - start_time
			print("--Mean loss: {} (time: {}s) --Train mAP (max 1.0): {} (time: {}s)".format(sum(mean_loss)/len(mean_loss), end_loss_diff, mean_avg_prec, end_epoch_diff))
			#tensorboard
			writer.add_scalar('Mean Loss/train', sum(mean_loss)/len(mean_loss), epoch ) #* len(train_loader) + batch_idx*BATCH_SIZE
			writer.add_scalar('mAP/train', mean_avg_prec, epoch ) #* len(train_loader) + batch_idx*BATCH_SIZE
		else:
			print("--Mean loss: {} (coord: {}; obj: {}; no_obj: {}; class: {}) (time: {}s)".format(sum(mean_loss)/len(mean_loss)
				, sum(mean_box_loss)/len(mean_box_loss)
				, sum(mean_object_loss)/len(mean_object_loss)
				, sum(mean_no_object_loss)/len(mean_no_object_loss)
				, sum(mean_class_loss)/len(mean_class_loss)
				, end_loss_diff))
			#tensorboard
			writer.add_scalar('Mean Loss/train', sum(mean_loss)/len(mean_loss), epoch ) #* len(train_loader) + batch_idx*BATCH_SIZE

	#calculate mAP of the model (test set)
	pred_boxes, target_boxes = get_bboxes(test_loader, model, iou_threshold=0.5, threshold=0.4, split_size=SPLIT_SIZE,anchor_count = NUM_BOXES, num_classes = NUM_CLASSES, anchors=anchors, device=DEVICE)
	mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
	end_total_time = time.time()
	total_time = end_total_time - start_total_time
	print("[TEST] --Test mAP (max 1.0): {} (total time: {}s)".format(sum(mean_loss)/len(mean_loss), end_loss_diff, mean_avg_prec, end_epoch_diff))

	#save final model into file
	if (SAVE_MODEL):
		checkpoint = {
			"state_dict": model.state_dict(),
			"optimizer": optimizer.state_dict(),
		}
		save_checkpoint(checkpoint, filename=SAVE_MODEL_FILE)
	writer.close() #close logger

if __name__ == "__main__":
	main()