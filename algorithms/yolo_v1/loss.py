import torch
import torch.nn as nn
from utils import intersection_over_union

class yolo_v1_loss(nn.Module):
	def __init__(self, split_size=7, num_boxes=2, num_classes=20):
		super().__init__()
		self.mse = nn.MSELoss(reduction="sum")

		self.split_size = split_size
		self.num_boxes = num_boxes
		self.num_classes = num_classes
		#yolo loss coeficcients (constants used to prioritize/de-preoritize certain parts of loss function)
		self.lambda_noobj = 0.5 #no-object in cell loss
		self.lambda_coord = 5 #coordinate loss

	# @param predictions all bounding boxes for all grid cells , tensor.shape = [x,S*S*(B*5+C)], WHERE x =num_samples, S= grid size, B= bbox count, C=num classes (5= x,y,w,h and obj prob)
	# @param target tensor.shape = (x, S, S, B*5+C) , WHERE: x=num samples, S= grid size, B= bbox count, C=num classes
	def forward(self, predictions, target):
		# reshape prediction from (BATCH_SIZE, S*S(C+B*5) -> (BATCH_SIZE, split_size , split_size, num_classes + num_boxes *5) eg. (1470) -> (7,7,30)
		predictions = predictions.reshape(-1, self.split_size, self.split_size, self.num_classes + self.num_boxes * 5)

		# calculate IoU of each predicted box with ground truth box
		# for Yolo_v1 num_boxes = 2 , so Calculate IoU for the two predicted bounding boxes with target bbox
		# (those occupy two porition of the re-shaped tensor), each box = [w,h,x,y, probability_of_object]
		iou_box_1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25]) # (x,S,S, 1) - IoU per each cell (1st bounding box of the cell)
		iou_box_2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25]) # (x,S,S, 1) - IoU per each cell (2nd bounding box of the cell)
		# print(target.shape)
		# iou_box_2[0][1][1] = 1
		# print(iou_box_1)
		# print(iou_box_2)
		iou_joined = torch.cat([iou_box_1.unsqueeze(0), iou_box_2.unsqueeze(0)], dim=0) #concat into one tensor (2, S,S,1)

		# accroding to yolo algorithm: out of two boxes predicted - take the box with highest IoU with ground truth
		# iou_max - holds max of values per each cell out of all IoU calculated there; iou_max_index - hold index values (ie. 0, 1... until bbox count per cell)  
		# the value stored in iou_max_index would relate to first dimention (ie. dim=0) of "iou_joined"
		iou_max, iou_max_index = torch.max(iou_joined, dim=0) 
		# target last dimention is B*5+C (e.g. 2*5+20=30) and contains (0..C-1, Pr(obj),X,Y,W,H, Pr(obj),X2,Y2, W2,H2 ...),  in case of groudn truth Pr(obj) is 0 or 1
		# we check only existance of one bbox in ground truth (yolo_v1 has limitation to 1 object per cell, so the final 5 numbers( ie. 26..30) are empty) )
		exists_box = target[..., 20].unsqueeze(3)  # (in paper this is identity function(obj_i) in loss equation) =1 if there is objected in cell, =0 if not
		# calculate loss components:
		# (components 1,2,3 - penalize only bounding box prediction with highest IoU with ground truth; 3b - penalize all bboxes predictions; 4 - class probabilities independent of bbox)

		# 1. coord (bbox) loss
		# ---------------------

		# upfront combine box1 and box2 into one tensor, pick the the value based on which IoU is higher; shape= (x, S, S, 2) ; 2 = x,y
		# (in the paper this is done with I(obj)ij identity function )
		box_xy_predictions = exists_box * (
			(
				iou_max_index * predictions[..., 26:28]
				+ (1 - iou_max_index) * predictions[..., 21:23]
			)
		)

		# pepare targets (make sure the coords for calls that have no bbox are zeroed) ; shape = (x,S,S,2)
		box_xy_targets = exists_box * target[..., 21:23]


		# TODO: remove this commented block
		# calculate squared error across all cells (like in the paper)
		# flatten tensors so all grid cells for all images are in single dimention (in order to fit the nn.MSE function arguments)
		# end_dim=-2 == keep the last dimention (ie. xy coords)
		# torch.flatten.shape = (x*S*S,2)
		# bbox_coord_loss.shape = (1)  - single number donoting error
		# bbox_coord_loss = self.mse(
		# 	torch.flatten(box_xy_predictions, end_dim=-2),
		# 	torch.flatten(box_xy_targets, end_dim=-2),
		# )
		#note: non-flatten gives same result

		#calculate squared error across all cells (like in the paper)
		bbox_coord_loss = self.mse(
			box_xy_predictions,
			box_xy_targets,
		)
		
		# 2. size (bbox) loss
		# ---------------------

		# upfront combine box1 and box2 into one tensor, pick the the value based on which IoU is higher; shape= (x, S, S, 2) ; 2 = w,h
		# (in the paper this is done with I_obj(ij) identity function )
		box_size_predictions = exists_box * (
			(
				iou_max_index * predictions[..., 28:30]
				+ (1 - iou_max_index) * predictions[..., 23:25]
			)
		)
		# pepare targets (make sure the coords for calls that have no bbox are zeroed) ; shape = (x,S,S,2)
		box_size_targets = exists_box * target[..., 23:25]

		# Additionally for size component we do sqrt prior to sum (as in the paper)  // "1e-6" added o avoid zero; abs to assure no sqrt from negative
		box_size_predictions[..., 0:2] = torch.sign(box_size_predictions[..., 0:2]) * torch.sqrt(torch.abs(box_size_predictions[..., 0:2] + 1e-6))
		box_size_targets[..., 0:2] = torch.sqrt(box_size_targets[..., 0:2])
		#TEMP: when random init values can go below zero so need to protect against that too (in production bounding boxes sizes are always >0)
		#box_size_targets[..., 0:2] = torch.sign(box_size_targets[..., 0:2]) * torch.sqrt(torch.abs(box_size_targets[..., 0:2] + 1e-6))

		# calculate squared error across all cells (like in the paper)
		bbox_size_loss =  self.mse(
			box_size_predictions,
			box_size_targets,
		)

		# 3. object loss (exist obj)
		# ---------------------
		# upfront combine box1 and box2 into one tensor, pick the the value based on which IoU is higher; shape= (x, S, S, 1) ; 1 = object probability in cell
		# (in the paper this is done with I_obj(ij) identity function )
		obj_prob_predictions = exists_box * (
			(
				iou_max_index * predictions[..., 25:26]
				+ (1 - iou_max_index) * predictions[..., 20:21]
			)
		)

		# pepare targets (make sure the coords for calls that have no bbox are zeroed) ; shape = (x,S,S,2)
		obj_prob_targets = exists_box * target[..., 20:21]

		# calculate squared error across all cells (like in the paper)
		obj_loss =  self.mse(
			obj_prob_predictions,
			obj_prob_targets,
		)
		# 3b. object loss (no exist obj)
		# (this one is not clear in paper what to take, implemented here: penalize both predicted bounding boxes )
		# ------------------------------

		# bbox1, calculate squared error across all cells (like in the paper)
		# (1 - exists_box) == zeroes out prediction error in cells that have an object (already accounted in previous loss component)
		no_obj_loss =  self.mse(
			(1 - exists_box) * predictions[..., 20:21],
			(1 - exists_box) * target[..., 20:21],
		)

		# bbox2 (add error), calculate squared error across all cells (like in the paper)
		no_obj_loss +=  self.mse(
			(1 - exists_box) * predictions[..., 25:26],
			(1 - exists_box) * target[..., 20:21],
		)
		# 4. class loss
		# -------------
		class_loss =  self.mse(
			exists_box * predictions[..., 0:20],
			exists_box * target[..., 0:20],
		)

		# sum all loss components (multypling with weights)
		total_loss = (
			self.lambda_coord * bbox_coord_loss  	# box coordinate loss
			+ self.lambda_coord * bbox_size_loss  	# box size loss
			+ obj_loss  							# object loss
			+ self.lambda_noobj * no_obj_loss	  	# no-object loss
			+ class_loss  							# class loss
		)

		return total_loss

# def test():
# 	from model import YOLO_v1
# 	model = YOLO_v1(in_channels=3, split_size=7, num_boxes=2, num_classes=20)
# 	x = torch.randn((1,3,448,448)) #simulate two images 448x448, 3 channels
# 	#x = torch.randn((2,3,400,400))
# 	out = model(x)
# 	#test loss
# 	y = torch.randn((1, 7, 7, 30)) #generate random ground truth tensor (x,S,S,B*5+C) WHERE: x=num samples, S= grid size, B= bbox count, C=num classes
# 	loss_fn = yolo_v1_loss()
# 	loss = loss_fn(out, y) #calculate loss
# 	print(loss)

# test()
