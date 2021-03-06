import torch
import os
import pandas as pd
from PIL import Image

# PyTorch Dataset class for Pascal VOC (2007+1012) converted into a custom format (ie. custom directory structure and custom annotations files)
# (the custom VOC format is more suitable for input data used by Yolo_v1, conversion files done by Aladdin Persson, part of his "from scrach" series))
# YOLO_v1 algorithm assumtions:
# yolo_v1 splits image into a grid split_size x split_size, each cell can have num_boxes and num_classes 
# (in practice yolo_v1 limits num_boxes to 1 in ground truth and to 2 for predictions generated by network. Here for sake of keeping both tensors same size we use "2" for grouth truth, second will be filled with zeroes)
class VOCDataset_custom(torch.utils.data.Dataset):

	# @param filelist_csv_path - csv file with pairs (image_filename, annotation_filename)
	# @param image_dir path to directory with images
	# @param annotation_dir path to directory with labels (ie bounding boxes and classes)
	# @param split_size (S) = grid size ( split_size x split_size cells)
	# @param num_boxes (B) = bounding boxes per cell
	# @param num_classes (C) = amount of classes 
	# @param transform transformation to b applied to image (ie. augmentations)
	def __init__(self, filelist_csv_path = "data/filelist.csv", image_dir = "data/images/", annotation_dir = "data/labels/", transform=None, split_size=7, num_boxes=2, num_classes=20):
		self.filelist_csv = pd.read_csv(filelist_csv_path)
		self.image_dir = image_dir
		self.annotation_dir = annotation_dir
		self.transform = transform
		#yolo specific paraeters
		self.split_size = split_size
		self.num_boxes = num_boxes
		self.num_classes = num_classes

	def __len__(self):
		return len(self.filelist_csv)


	# for every image need to convert annotations (x,y,w,h) from full image relative to grid cell relative (this is how yolo algorithm processes it)
	# @param index index of the image/annotation to extract
	def __getitem__(self, index):
		#load annotations
		annotation_file_path = os.path.join(self.annotation_dir, self.filelist_csv.iloc[index, 1])
		bounding_boxes = []
		#1. read annotations file line by line and extract bounding box data
		with open(annotation_file_path) as annotation_file:
			for annotation in annotation_file.readlines():
				annotation_split = annotation.replace("\n", "").split()

				class_label = int(annotation_split[0])
				x_centre = float(annotation_split[1])
				y_centre = float(annotation_split[2])
				width = float(annotation_split[3])
				height = float(annotation_split[4])

				bounding_boxes.append([class_label, x_centre, y_centre, width, height])

		#bounding_boxes = torch.tensor(bounding_boxes)

		#2. read image
		image_path = os.path.join(self.image_dir, self.filelist_csv.iloc[index, 0])
		image = Image.open(image_path)

		#3. apply transformations is any (both image and bounding boxes)
		if self.transform:
			image, bounding_boxes = self.transform(image, bounding_boxes)

		#4. convert annotations from image relative to grid cell relative
		# init empty tensor that will store all annotations per each grid cell in the image , shape = (S, S, C + B*5) ; final dimention of that tensor contains:
		# - annotations_tensor[i,j,0:C] =classes (annotations_tensor[x] = 1, denotes object beloging to class x, all other will be zeroes)
		# - annotations_tensor[i,j,20] = if object exists in cell
		# - annotations_tensor[i,j,21:23] = x,y coords
		# - annotations_tensor[i,j,23:25] = width,height coords
		annotations_tensor = torch.zeros((self.split_size, self.split_size, self.num_classes + 5 * self.num_boxes)) 

		#fill out tensor with data earlier read from annotations file
		for bounding_box in bounding_boxes:
			class_label, x_centre, y_centre, width, height = bounding_box

			#calculate new x,y coordinates
			cell_row, cell_col = int(self.split_size * y_centre), int(self.split_size * x_centre) #calculate which cell the annotation belogs to (total size is split_size x split_size)
			x_centre_cell, y_centre_cell = self.split_size * x_centre - cell_col, self.split_size * y_centre - cell_row #calculate coordinate inside the cell

			#caluclate new width/height relative to cell size (ie. cell is split_size times small than image, so need to multiply to have same width in new coord system)
			width_cell, height_cell = (
				width * self.split_size,
				height * self.split_size,
			)

			#set the class and append x,y,width,heigth
			if 	annotations_tensor[cell_row, cell_col, 20] == 0: #yolo_v1 limits to only 1 bounding box per cell, so set only if there wasnt an object prior in the cell
				#set all cell data
				annotations_tensor[cell_row, cell_col, 20] = 1 # object probability = 1
				annotations_tensor[cell_row, cell_col, 21] = x_centre_cell
				annotations_tensor[cell_row, cell_col, 22] = y_centre_cell
				annotations_tensor[cell_row, cell_col, 23] = width_cell
				annotations_tensor[cell_row, cell_col, 24] = height_cell
				annotations_tensor[cell_row, cell_col, class_label] = 1 # set class
				
		return image, annotations_tensor

# def test():

# 	dataset = VOCDataset_custom(filelist_csv_path= "data_small/8examples.csv"
# 								, image_dir = "data_small/images/"
# 								, annotation_dir = "data_small/labels/"
# 								, transform=None
# 								, split_size=7, num_boxes=2, num_classes=20)

# 	#x = torch.randn((2,3,400,400))
# 	image, annotations_tensor = dataset.__getitem__(0)
# 	print(annotations_tensor.shape)
# 	print("----------------------")
# 	torch.set_printoptions(profile="full")
# 	print(annotations_tensor)
# 	torch.set_printoptions(profile="default")

# test()
		