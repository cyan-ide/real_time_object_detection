import torch
import torch.nn as nn
from utils import intersection_over_union

class YOLOv2_loss(nn.Module):
	def __init__(self, split_size=13, num_boxes=5, num_classes=20, anchors=None):
		super().__init__()
		self.mse = nn.MSELoss(reduction="sum")
		#self.mse = nn.MSELoss()
		self.bce = nn.BCEWithLogitsLoss()
		self.entropy = nn.CrossEntropyLoss()
		self.sigmoid = nn.Sigmoid()

		self.split_size = split_size
		self.num_boxes = num_boxes
		self.num_classes = num_classes
		self.anchors = anchors

		#reshape anchor tensor to match predictions (batch_size, NUM_ANCHORS, split_size, split_size, 2) , 2 = width and height from anchor
		self.anchors = self.anchors.reshape(1, self.num_boxes, 1, 1, 2) 

		#yolo loss coeficcients (constants used to prioritize/de-preoritize certain parts of loss function)
		self.lambda_box = 10 #10	# coordinate / box size loss
		self.lambda_obj = 100		# object in cell loss   	
		self.lambda_noobj = 0.5 # no-object in cell loss
		self.lambda_class = 1	# class prediction loss

		# self.lambda_box = 1.0 #10	# coordinate / box size loss
		# self.lambda_obj = 5.0		# object in cell loss   	
		# self.lambda_noobj = 1.0 # no-object in cell loss
		# self.lambda_class = 1.0	# class prediction loss
		

	# @param predictions all bounding boxes for all grid cells , tensor.shape = [x,S*S*(B*5+C)], WHERE x =num_samples, S= grid size, B= bbox count, C=num classes (5= x,y,w,h and obj prob)
	# @param target tensor.shape = (x, S, S, B*5+C) , WHERE: x=num samples, S= grid size, B= bbox count, C=num classes
	def forward(self, predictions, target): 
		#batch size
		batch_size = target.shape[0]
		# reshape prediction from (BATCH_SIZE, (C+5)*B,S,S ) -> (BATCH_SIZE, split_size , split_size, num_classes + num_boxes *5) eg. (125,13,13) -> (5,13,13,25)
		predictions = predictions.reshape(-1, self.num_boxes, self.num_classes + 5, self.split_size, self.split_size)
		predictions = predictions.permute(0,1,3,4,2)

		# Check where obj and noobj (we ignore if target == -1)
		obj = target[..., self.num_classes] == 1  # in paper this is Iobj_i
		noobj = target[..., self.num_classes] == 0  # in paper this is Inoobj_i


		# calculate loss components:
		# (components 1,2,3 - penalize only bounding box prediction with highest IoU with ground truth; 3b - penalize all bboxes predictions; 4 - class probabilities independent of bbox)

		# 1a. object loss (exist obj)
		# ---------------------
		
		#calculate IoU between predicted box and target box (x/y coords and width/height)
		box_preds = torch.cat([self.sigmoid(predictions[..., self.num_classes+1:self.num_classes+3]), torch.exp(predictions[..., self.num_classes+3:self.num_classes+5]) * self.anchors], dim=-1) #transform predicted offsets to YOLOv2 coords, width,height
		ious = intersection_over_union(box_preds[obj], target[..., self.num_classes+1:self.num_classes+5][obj]).detach() #calculate IoU

		#calculate mean squered error between predicted "object probability" and target probability 
		#(again [obj] is the identify function that includes only those calculations where ground truth exists)
		# additionally the target score is multiplied by IoU (otherwise it would just be "1")
		object_loss = self.mse(self.sigmoid(predictions[..., self.num_classes:self.num_classes+1][obj]), ious * target[..., self.num_classes:self.num_classes+1][obj]) #
		

		# 2. coord (bbox) loss
		# ---------------------
		#print("cuda test| target: {}, anchors: {}".format(target.is_cuda, self.anchors.is_cuda))
		
		# convert network bounding box prediction for x,y to sigmoid(x), sigmoid(y) , as in the paper b_x = sigmoid(t_x)
		predictions[..., self.num_classes+1:self.num_classes+3] = self.sigmoid(predictions[..., self.num_classes+1:self.num_classes+3])  # x,y coordinates # self.sigmoid(predictions[..., 1:3])

		# convert ground truth width / height into network output (ie. in paper target_w = anchor_w * exp(network_pred_w) -> here we do inverse because log safer to calculate?)
		target[..., self.num_classes+3:self.num_classes+5] = torch.log(
			(1e-16 + target[..., self.num_classes+3:self.num_classes+5] / self.anchors)
		)  # width, height coordinates

		#calculate mean squered error between predicted coords/width/height and targets 
		# [obj] picks only those predictions where corresponding target exists (rest is zeroes out/ not included in loss function result)
		box_loss = self.mse(predictions[..., self.num_classes+1:self.num_classes+5][obj], target[..., self.num_classes+1:self.num_classes+5][obj])

		# 1b. object loss (no exist obj)
		# ------------------------------
		# similar as object loss (except not taking IoU anymore for target)
		# also instead of [obj] identify function using [noobj] (which means that here we add to the error all predictions where target object does not exist)
		# this is done seperatly from "2a. object loss (exist obj)" as there are different weights for penalizing lack of prediction where it should be; and presence of prediction where it shouldnt exist

		no_object_loss = self.mse(
			self.sigmoid(predictions[..., self.num_classes:self.num_classes+1][noobj]), (target[..., self.num_classes:self.num_classes+1][noobj]),
		)

		# 3. class loss
		# -------------

		class_loss =  self.mse(
			predictions[..., 0:self.num_classes][obj],
			target[..., 0:self.num_classes][obj],
		)

		# sum all loss components (multypling with weights)
		total_loss = (
			self.lambda_box * box_loss  			# includes box coordinate loss and box size loss
			+ self.lambda_obj * object_loss			# object loss
			+ self.lambda_noobj * no_object_loss  	# no-object loss
			+ self.lambda_class * class_loss		# class loss
		)

		return total_loss, self.lambda_box * box_loss, self.lambda_obj * object_loss, self.lambda_noobj * no_object_loss, self.lambda_class * class_loss

def test():
	from model import YOLO_v2
	from darknet_model import Darknet19, ResidualBlockStart

	DEVICE = "cuda"

	anchors = torch.tensor( [(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)] ).to(DEVICE)

	darknet = Darknet19(classification_head = False).to(DEVICE)
	model = YOLO_v2(darknet_layers = darknet.darknet_layers, in_channels=3, split_size=13, num_boxes=5, num_classes=20).to(DEVICE)
	x = torch.randn((2,3,416,416)).to(DEVICE) #simulate two images 448x448, 3 channels
	out = model(x)
	y = torch.randn((2, 5, 13, 13, 25)) #generate random ground truth tensor (x,S,S,B*5+C) WHERE: x=num samples, S= grid size, B= bbox count, C=num classes


	y[...,20] = 0 #set probability of bounding box to zero for all
	# set some fake bounxing boxes
	y[1,4,12,12,20] = 1
	y[1,4,6,6,20] = 1
	y[0,2,2,3,20] = 1
	y[0,3,3,4,20] = 1
	y = abs(y).to(DEVICE) #make sure no values < zero, like negative width or height, as loss calculates log (which would turn into "nan" if < 0)
	
	#custom output/ ground truth
	out_fake = torch.zeros(1,125,13,13, requires_grad=False)
	# second dimention per each anchor 25 numbers = [0..num_clases, obj_prob, x, y, w, h]
	# 0 = first anchor

	import numpy as np
	import math
	out_fake[0,0*25 + 5,1,1] =  1 # class 5
	out_fake[0,0*25 + 20,1,1] =  np.log(0.9999/(1-0.9999))  # logit(obj probability) 
	out_fake[0,0*25 + 21,1,1] =  np.log(0.1/(1-0.1)) #0.1 # x = logit(ground_truth_x)
	out_fake[0,0*25 + 22,1,1] =  np.log(0.3/(1-0.3)) #0.3 # y = logit(ground_truth_y)
	out_fake[0,0*25 + 23,1,1] =  np.log(1e-16 + 0.4 / 1.3221) #0.4 # w = torch.log( (1e-16 + y_w / anchor_w)
	out_fake[0,0*25 + 24,1,1] =  np.log(1e-16 + 0.6 / 1.73145) #0.6 # h
	out_fake=out_fake.to(DEVICE)

	y_fake = torch.zeros((1, 5, 25, 13, 13)) #generate random ground truth tensor (x,S,S,B*5+C) WHERE: x=num samples, S= grid size, B= bbox count, C=num classes

	# set some fake bounxing boxes
	# y_fake[0,0,5,1,1] = 1
	# y_fake[0,0,20,1,1] = 1
	# y_fake[0,0,21,1,1] = 0.2
	# y_fake[0,0,22,1,1] = 0.3
	# y_fake[0,0,23,1,1] = 0.4
	# y_fake[0,0,24,1,1] = 0.6
	# y_fake=y_fake.to(DEVICE)

	y_fake = torch.zeros((1, 5, 13, 13, 25))
	y_fake[0,0,1,1,5] = 1 #class
	y_fake[0,0,1,1,20] = 1 #obj prob
	y_fake[0,0,1,1,21] = 0.1 #1 / (1 + math.exp(-0.1)) #0.2 #x = sigmoid(x_out)
	y_fake[0,0,1,1,22] = 0.3 #1 / (1 + math.exp(-0.3)) #0.3 #y
	y_fake[0,0,1,1,23] = 0.4 #w
	y_fake[0,0,1,1,24] = 0.6 #h
	y_fake=y_fake.to(DEVICE)

	print("Model input shape: {}".format(x.shape))
	print("Model output shape: {}".format(out.shape))
	print("Model output(fake) shape: {}".format(out_fake.shape))
	print("Ground truth shape: {}".format(y.shape))

	#test loss
	loss_fn = YOLOv2_loss(anchors=anchors)
	#loss = loss_fn(out, y) #calculate loss
	# print("out_fake, before: {}".format(out_fake[0,0:25,1,1]))
	loss = loss_fn(out_fake, y_fake) #calculate loss
	print("loss (1nd): {}".format(loss))


	#2nd iter
	out_fake = torch.zeros(1,125,13,13, requires_grad=False)
	out_fake[0,0*25 + 5,1,1] =  1 # class 5
	out_fake[0,0*25 + 20,1,1] =  np.log(0.9999/(1-0.9999))  # logit(obj probability) 
	out_fake[0,0*25 + 21,1,1] =  np.log(0.1/(1-0.1)) #0.1 # x = logit(ground_truth_x)
	out_fake[0,0*25 + 22,1,1] =  np.log(0.3/(1-0.3)) #0.3 # y = logit(ground_truth_y)
	out_fake[0,0*25 + 23,1,1] =  np.log(1e-16 + 0.4 / 1.3221) #0.4 # w = torch.log( (1e-16 + y_w / anchor_w)
	out_fake[0,0*25 + 24,1,1] =  np.log(1e-16 + 0.6 / 1.73145) #0.6 # h

	out_fake[0,1*25 + 20,1,1] =  np.log(0.9999/(1-0.9999))  # logit(obj probability) 
	out_fake=out_fake.to(DEVICE)

	y_fake = torch.zeros((1, 5, 13, 13, 25))
	y_fake[0,0,1,1,5] = 1 #class
	y_fake[0,0,1,1,20] = 1 #obj prob
	y_fake[0,0,1,1,21] = 0.1 #1 / (1 + math.exp(-0.1)) #0.2 #x = sigmoid(x_out)
	y_fake[0,0,1,1,22] = 0.3 #1 / (1 + math.exp(-0.3)) #0.3 #y
	y_fake[0,0,1,1,23] = 0.4 #w
	y_fake[0,0,1,1,24] = 0.6 #h
	y_fake=y_fake.to(DEVICE)
	loss = loss_fn(out_fake, y_fake) #calculate loss
	print("loss (2nd): {}".format(loss))


	print("out_fake, after: {}".format(out_fake[0,0:25,1,1]))

# test()
