import torch
import torch.nn as nn
import torch.optim as optim

from darknet_model import Darknet19, ResidualBlockStart #for testing only
from torchsummary import summary
import time

#imapge channekls
# in_channels = 3

#wrapper around regular Module to indicate Residual Block end
#(Residual Block is split into Start/Stop classes because Darknet architecture has only the starting point of the block, the ending is in YOLO_v2)
class ResidualBlockEnd(nn.Module):
	def __init__(self, module):
		super(ResidualBlockEnd, self).__init__()
		self.module = module

	def forward(self, x):
		return self.module(x)

# 2. network architecture
class YOLO_v2(nn.Module):
	
	# darknet_layers - darknet neural network layers
	# in_channels = amount of channels of image (e.g. 3 = RGB, 1 = black and white)
	# split_size (S) = grid size ( split_size x split_size cells)
	# num_boxes (B) = bounding boxes per cell
	# num_classes (C) = amount of classes 
	def __init__(self, darknet_layers, in_channels=3, split_size=13, num_boxes=5, num_classes=20):
		super().__init__()

		#define network elements
		#layers = []
		layers = nn.ModuleList()
		layers.extend(darknet_layers)
		# --- darknet ends here (above pre-trained on ImageNet) ---
		layers.append( nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False) ) #in channels equal to output of prior
		layers.append( nn.BatchNorm2d(num_features=1024) ) #input equal to output size of prior layer
		layers.append( nn.LeakyReLU(0.1) )
		# 
		layers_sequence = []
		layers_sequence += [ nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False) ] #in channels equal to output of prior
		layers_sequence += [ nn.BatchNorm2d(num_features=1024) ] #input equal to output size of prior layer
		layers_sequence += [ nn.LeakyReLU(0.1) ]
		layers.append( ResidualBlockEnd( nn.Sequential(*layers_sequence) ) ) #output of this layer is later used in YOLOv2 architecture
		# <Residual Block ends above> the input here will be concat with earlier layer from Darknet
		layers.append( nn.Conv2d(in_channels=3072, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False) ) #in channels equal to output of prior
		layers.append( nn.BatchNorm2d(num_features=1024) ) #input equal to output size of prior layer
		layers.append( nn.LeakyReLU(0.1) )
		# # # final detection layer
		layers.append( nn.Conv2d(in_channels=1024, out_channels=num_boxes * (5+num_classes), kernel_size=1, stride=1, padding=0, bias=False) ) #in channels equal to output of prior
		# output has 13x13 resoution feature maps, which is equivalent to 13x13 split_size applied by yolo to detect bounding boxes
		# each channel of output encodes a different anchor box data paramter (in YOLOv1 all was put in a single channel of fully connected layer )
		# output channel count is equal to : bounding_box_count_per_single_cell * (5 + class_count)
		# "5" = width,height, xCenter,yCenter, obj_probability

		self.yolo_v2 = layers #= nn.Sequential(*layers)
		
	# Reorganization layer part of YOLOv2 passthrough from final 512 layer in Darknet19, to the last but one Conv layer in YOLOv2 network
	# the "reorg" downsamples the signal and additionally "reorders the pixels": the neighbouring pixels are stacked into additional layers
	# (e.g. 26x26 resolution changes to 13x13, the downsamples pixels are put into additional layers)
	# transform input 26x26x512 -> 13x13x2048
	def reorg_layer(self, x):
		stride = 2
		batch_size, channels, height, width = x.size()
		new_ht = int(height/stride)
		new_wd = int(width/stride)
		new_channels = channels * stride * stride
		passthrough = x.permute(0, 2, 3, 1)
		passthrough = passthrough.contiguous().view(-1, new_ht, stride, new_wd, stride, channels)
		passthrough = passthrough.permute(0, 1, 3, 2, 4, 5)
		passthrough = passthrough.contiguous().view(-1, new_ht, new_wd, new_channels)
		passthrough = passthrough.permute(0, 3, 1, 2)
		return passthrough

	# @param _input image 448x448, tensor size = (x,3,448,448) = (x,N_CHANNEL,WIDTH,HEIGHT) , WHERE: x=num samples, N_CHANNEL = no channels in image (eg RGB =3), WIDTH, HEIGHT = image dimentions
	def forward(self, input_):
		#run the forward pass  (ie. link blocks defined in the init)

		residualBlockPassthrough = []
		x = input_
		for layer in self.yolo_v2:
			x = layer(x) #pass output of previous layer through next layer

			#special behaviour for residual block (in YOLOv2 this is called "passthrough")
			if isinstance(layer, ResidualBlockStart):
				residualBlockPassthrough = [x] #record layer output at the start of the block
			if isinstance(layer, ResidualBlockEnd):
				passthroughDownsampled = self.reorg_layer( residualBlockPassthrough.pop() ) #prior to concat have to transform to match the current downsampled output
				x = torch.cat([passthroughDownsampled, x], 1) #concat output recorded before with current layer output at the end of the block
		return x

def test():
	DEVICE = "cuda"
	darknet = Darknet19(classification_head = False).to(DEVICE)
	print("-----DARKNET19 (in: 224x224)-----")
	summary(darknet, (3, 224, 224))
	print("-----DARKNET19 (in: 448x448)-----")
	summary(darknet, (3, 448, 448))
	print("-----DARKNET19 (in: 416x416)-----")
	summary(darknet, (3, 416, 416))
	print("-----YOLOv2 (in: 224x224)-----")
	model = YOLO_v2(darknet_layers = darknet.darknet_layers, in_channels=3, split_size=13, num_boxes=5, num_classes=20).to(DEVICE)
	#darknet high/lo res
	summary(model, (3, 224, 224))
	print("-----YOLOv2 (in: 416x416)-----")
	model = YOLO_v2(darknet_layers = darknet.darknet_layers, in_channels=3, split_size=13, num_boxes=5, num_classes=20).to(DEVICE)
	#darknet high/lo res
	summary(model, (3, 416, 416))
	print("---------------------------------")
	#yolo train size
	x = torch.randn((8,3,416,416)).to(DEVICE)
	start1 = time.time()
	out = model(x)
	end1 = time.time() - start1
	print("time: {}s".format(end1))


# test()
