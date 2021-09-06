import time

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import torchvision.ops

#import time

def iou_width_height(boxes1, boxes2):
	"""
	Parameters:
		boxes1 (tensor): width and height of the first bounding boxes
		boxes2 (tensor): width and height of the second bounding boxes
	Returns:
		tensor: Intersection over union of the corresponding boxes
	"""
	intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
		boxes1[..., 1], boxes2[..., 1]
	)
	union = (
		boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
	)
	return intersection / union


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
	"""
	Calculates intersection over union

	Parameters:
		boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
		boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
		box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

	Returns:
		tensor: Intersection over union for all examples
	"""

	if box_format == "midpoint":
		box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
		box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
		box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
		box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
		box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
		box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
		box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
		box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

	if box_format == "corners":
		box1_x1 = boxes_preds[..., 0:1]
		box1_y1 = boxes_preds[..., 1:2]
		box1_x2 = boxes_preds[..., 2:3]
		box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
		box2_x1 = boxes_labels[..., 0:1]
		box2_y1 = boxes_labels[..., 1:2]
		box2_x2 = boxes_labels[..., 2:3]
		box2_y2 = boxes_labels[..., 3:4]

	x1 = torch.max(box1_x1, box2_x1)
	y1 = torch.max(box1_y1, box2_y1)
	x2 = torch.min(box1_x2, box2_x2)
	y2 = torch.min(box1_y2, box2_y2)

	# .clamp(0) is for the case when they do not intersect
	intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

	box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
	box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

	return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
	"""
	Does Non Max Suppression given bboxes

	Parameters:
		bboxes (list): list of lists containing all bboxes with each bboxes
		specified as [class_pred, prob_score, x1, y1, x2, y2]
		iou_threshold (float): threshold where predicted bboxes is correct
		threshold (float): threshold to remove predicted bboxes (independent of IoU) 
		box_format (str): "midpoint" or "corners" used to specify bboxes

	Returns:
		list: bboxes after performing NMS given a specific IoU threshold
	"""

	assert type(bboxes) == list

	bboxes = [box for box in bboxes if box[1] > threshold]
	bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
	bboxes_after_nms = []

	while bboxes:
		start3 = time.time()
		chosen_box = bboxes.pop(0)

		
		bboxes = [
			box
			for box in bboxes
			if box[0] != chosen_box[0]
			or intersection_over_union(
				torch.tensor(chosen_box[2:]),
				torch.tensor(box[2:]),
				box_format=box_format,
			)
			< iou_threshold
		]
		

		bboxes_after_nms.append(chosen_box)

	return bboxes_after_nms


def mean_average_precision(
	pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
	"""
	Calculates mean average precision 

	Parameters:
		pred_boxes (list): list of lists containing all bboxes with each bboxes
		specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
		true_boxes (list): Similar as pred_boxes except all the correct ones 
		iou_threshold (float): threshold where predicted bboxes is correct
		box_format (str): "midpoint" or "corners" used to specify bboxes
		num_classes (int): number of classes

	Returns:
		float: mAP value across all classes given a specific IoU threshold 
	"""

	# list storing all AP for respective classes
	average_precisions = []

	# used for numerical stability later on
	epsilon = 1e-6

	for c in range(num_classes):
		detections = []
		ground_truths = []

		# Go through all predictions and targets,
		# and only add the ones that belong to the
		# current class c
		for detection in pred_boxes:
			if detection[1] == c:
				detections.append(detection)

		for true_box in true_boxes:
			if true_box[1] == c:
				ground_truths.append(true_box)

		# find the amount of bboxes for each training example
		# Counter here finds how many ground truth bboxes we get
		# for each training example, so let's say img 0 has 3,
		# img 1 has 5 then we will obtain a dictionary with:
		# amount_bboxes = {0:3, 1:5}
		amount_bboxes = Counter([gt[0] for gt in ground_truths])

		# We then go through each key, val in this dictionary
		# and convert to the following (w.r.t same example):
		# ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
		for key, val in amount_bboxes.items():
			amount_bboxes[key] = torch.zeros(val)

		# sort by box probabilities which is index 2
		detections.sort(key=lambda x: x[2], reverse=True)
		TP = torch.zeros((len(detections)))
		FP = torch.zeros((len(detections)))
		total_true_bboxes = len(ground_truths)
		
		# If none exists for this class then we can safely skip
		if total_true_bboxes == 0:
			continue

		for detection_idx, detection in enumerate(detections):
			# Only take out the ground_truths that have the same
			# training idx as detection
			ground_truth_img = [
				bbox for bbox in ground_truths if bbox[0] == detection[0]
			]

			num_gts = len(ground_truth_img)
			best_iou = 0

			for idx, gt in enumerate(ground_truth_img):
				iou = intersection_over_union(
					torch.tensor(detection[3:]),
					torch.tensor(gt[3:]),
					box_format=box_format,
				)

				if iou > best_iou:
					best_iou = iou
					best_gt_idx = idx

			if best_iou > iou_threshold:
				# only detect ground truth detection once
				if amount_bboxes[detection[0]][best_gt_idx] == 0:
					# true positive and add this bounding box to seen
					TP[detection_idx] = 1
					amount_bboxes[detection[0]][best_gt_idx] = 1
				else:
					FP[detection_idx] = 1

			# if IOU is lower then the detection is a false positive
			else:
				FP[detection_idx] = 1

		TP_cumsum = torch.cumsum(TP, dim=0)
		FP_cumsum = torch.cumsum(FP, dim=0)
		recalls = TP_cumsum / (total_true_bboxes + epsilon)
		precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
		precisions = torch.cat((torch.tensor([1]), precisions))
		recalls = torch.cat((torch.tensor([0]), recalls))
		# torch.trapz for numerical integration
		average_precisions.append(torch.trapz(precisions, recalls))

	return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes, output_filepath="",show_frame= False):
	"""Plots predicted bounding boxes on the image"""
	im = np.array(image)
	height, width, _ = im.shape
	# Create figure and axes
	if (show_frame):
		fig, ax = plt.subplots(1)
	else:
		#off axis
		my_dpi = 100
		fig = plt.figure(frameon=False, figsize=(width/my_dpi, height/my_dpi), dpi=my_dpi )
		ax = plt.Axes(fig, [0., 0., 1., 1.])
		#ax.set_axis_off()
		fig.add_axes(ax)

	# Display the image
	ax.imshow(im, aspect='auto')
	#ax.imshow(im)

	# box[0] is x midpoint, box[2] is width
	# box[1] is y midpoint, box[3] is height

	# Create a Rectangle potch
	for box in boxes:
		box = box[2:]
		assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
		upper_left_x = box[0] - box[2] / 2
		upper_left_y = box[1] - box[3] / 2
		rect = patches.Rectangle(
			(upper_left_x * width, upper_left_y * height),
			box[2] * width,
			box[3] * height,
			linewidth=1,
			edgecolor="r",
			facecolor="none",
		)
		# Add the patch to the Axes
		ax.add_patch(rect)

	if len(output_filepath) > 0:
		plt.savefig(output_filepath)
	else:
		plt.show()

	plt.close()

#convert xywh box format to xyxy (corner coords)
def xywh2xyxy(x):
	# Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
	y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
	#print(x)
	y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
	y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
	y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
	y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
	return y

#convert xywh box format to xyxy (corner coords) [also move from width, heigh independent format to values expressed in pixels]
def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
	# Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
	y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
	y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
	y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
	y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
	y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
	return y

#split bboxes list into seperate lists per each class, also filter out boxes by threshold
def get_boxes_per_class(bboxes,threshold):
	bboxes_out =  {}
	scores_out = {}
	for box in bboxes:
		if box[1] <= threshold: #skip all boxes below/equal to threshold
			continue
		if box[0] in bboxes_out.keys():
			bboxes_out[box[0]].append( box[2:] ) 
			scores_out[box[0]].append(box[1]) 
		else:
			bboxes_out[box[0]] = [ box[2:] ] 
			scores_out[box[0]] = [ box[1] ] 
	return bboxes_out, scores_out


def get_bboxes(
	loader,
	model,
	iou_threshold,
	threshold,
	split_size = 13,
	anchor_count = 5, 
	num_classes = 20,
	anchors = None, 
	pred_format="cells",
	box_format="midpoint",
	device="cuda",
):
	all_pred_boxes = []
	all_true_boxes = []
	

	# make sure model is in eval before get bboxes
	model.eval()
	train_idx = 0

	for batch_idx, (x, labels) in enumerate(loader):
		x = x.to(device)
		labels = labels.to(device)

		with torch.no_grad():
			predictions = model(x)

		batch_size = x.shape[0]
		predictions = predictions.reshape(-1, anchor_count, num_classes + 5, split_size, split_size)
		predictions = predictions.permute(0,1,3,4,2)

		true_bboxes = cells_to_bboxes(labels, anchors, S=split_size, is_preds=False, anchor_count = anchor_count, num_classes = num_classes) #cellboxes_to_boxes(labels)
		bboxes = cells_to_bboxes(predictions, anchors, S=split_size, is_preds=True, anchor_count = anchor_count, num_classes = num_classes)

		allboxes_cnt = 0
		obj_prob = []
		for idx in range(batch_size):

			#use torchvision nms implementation
			#split boxes by class
			torch_nms_s_time = time.time() 

			bbox_per_class, bbox_scores_per_class = get_boxes_per_class(bboxes[idx], threshold) #= [ for box in bboxes[idx] ]
			#run nms for each class
			nms_full_output = torch.Tensor()
			for class_id in bbox_per_class.keys():				
				nms_output = torchvision.ops.boxes.nms( boxes = torch.FloatTensor( xywh2xyxy(np.array(bbox_per_class[class_id])) ), scores= torch.FloatTensor( bbox_scores_per_class[class_id] ) , iou_threshold= iou_threshold ) #-0.1
				nms_full_output = torch.cat( [nms_full_output, nms_output], dim = 0)
				for nms_box in nms_output:
					box = bbox_per_class[class_id][nms_box]
					box.insert(0,bbox_scores_per_class[class_id][nms_box])
					box.insert(0,class_id)
					all_pred_boxes.append([train_idx] + bbox_per_class[class_id][nms_box])

			torch_nms_e_time = time.time()

			allboxes_cnt += len(bboxes[idx])

			for box in true_bboxes[idx]:
				# many will get converted to 0 pred
				if box[1] > threshold:
					all_true_boxes.append([train_idx] + box)

			train_idx += 1

	model.train()

	return all_pred_boxes, all_true_boxes

def convert_cellboxes(predictions, S=7):
	"""
	Converts bounding boxes output from Yolo with
	an image split size of S into entire image ratios
	rather than relative to cell ratios. Tried to do this
	vectorized, but this resulted in quite difficult to read
	code... Use as a black box? Or implement a more intuitive,
	using 2 for loops iterating range(S) and convert them one
	by one, resulting in a slower but more readable implementation.
	"""

	predictions = predictions.to("cpu")
	batch_size = predictions.shape[0]
	predictions = predictions.reshape(batch_size, 7, 7, 30)
	bboxes1 = predictions[..., 21:25]
	bboxes2 = predictions[..., 26:30]
	scores = torch.cat(
		(predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
	)
	best_box = scores.argmax(0).unsqueeze(-1)
	best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
	cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
	x = 1 / S * (best_boxes[..., :1] + cell_indices)
	y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
	w_y = 1 / S * best_boxes[..., 2:4]
	converted_bboxes = torch.cat((x, y, w_y), dim=-1)
	predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
	best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
		-1
	)
	converted_preds = torch.cat(
		(predicted_class, best_confidence, converted_bboxes), dim=-1
	)

	return converted_preds


def cellboxes_to_boxes(out, S=7):
	converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
	converted_pred[..., 0] = converted_pred[..., 0].long()
	all_bboxes = []

	for ex_idx in range(out.shape[0]):
		bboxes = []

		for bbox_idx in range(S * S):
			bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
		all_bboxes.append(bboxes)

	return all_bboxes

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
	print("=> Saving checkpoint")
	torch.save(state, filename)

def cells_to_bboxes(predictions, anchors, S, is_preds=True, anchor_count = 5, num_classes = 20):
	"""
	Scales the predictions coming from the model to
	be relative to the entire image such that they for example later
	can be plotted or.
	INPUT:
	predictions: tensor of size (N, anchor_count, S, S, num_classes+5)
	anchors: the anchors used for the predictions
	S: the number of cells the image is divided in on the width (and height)
	is_preds: whether the input is predictions or the true bounding boxes
	anchor_count: total amount of anchors per grid cell (in YOLOv2 paper = 5)
	OUTPUT:
	converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
					  object score, bounding box coordinates
	"""
	BATCH_SIZE = predictions.shape[0]
	num_anchors = len(anchors)
	box_predictions = predictions[..., num_classes+1:num_classes+5]

	if is_preds:
		anchors = anchors.reshape(1, anchor_count, 1, 1, 2) #fit anchor tensor to same shape as model output (BATCH_SIZE, ANCHOR_COUNT, SPLIT_SIZE, SPLIT_SIZE, 2), 2 = width / height of anchor
		#transform model predictions as they are offsets, not direct bbox coords/dimentions
		box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2]) #x ,y coords
		box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors #height/width

		scores = torch.sigmoid(predictions[..., num_classes:num_classes+1])
		best_class = predictions[..., :num_classes].argmax(-1).unsqueeze(-1)

	else:
		scores = predictions[..., num_classes:num_classes+1]
		best_class = predictions[..., :num_classes].argmax(-1).unsqueeze(-1)

	cell_indices = (
		torch.arange(S)
		.repeat(BATCH_SIZE, anchor_count, S, 1)
		.unsqueeze(-1)
		.to(predictions.device)
	)
	# - going back to pixel = multiply by box size (e.g. 1/32px)
	# - going back to relative dimentions as below (ie. normalized by img size), multiply by 1 / cell_count (e.g. 1/13 = 1 cell is 0.07692307692 of image)
	# -- x = 1/ S * (x_in_cell_offset + cell_column)
	x = 1 / S * (box_predictions[..., 0:1] + cell_indices) 
	y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
	w_h = 1 / S * box_predictions[..., 2:4]
	converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, anchor_count * S * S, 6)
	#converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1) #.reshape(BATCH_SIZE, anchor_count * S * S, 4)
	return converted_bboxes.tolist()

def load_checkpoint(checkpoint, model, optimizer):
	print("=> Loading checkpoint")
	model.load_state_dict(checkpoint["state_dict"])
	optimizer.load_state_dict(checkpoint["optimizer"])
