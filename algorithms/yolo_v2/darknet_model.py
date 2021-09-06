import torch
import torch.nn as nn
import torch.optim as optim


# class Squeeze(nn.Module):
# 	def __init__(self):
# 		super(Squeeze, self).__init__()

# 	def forward(self, x):
# 		return x.squeeze()

class Reshape(nn.Module):
	def __init__(self):
		super(Reshape, self).__init__()

	def forward(self, x):
		return x.reshape(-1,1000) #squeeze()

#wrapper around regular Module to indicate Residual Block start
#(Residual Block is split into Start/Stop classes because Darknet architecture has only the starting point of the block, the ending is in YOLO_v2)
class ResidualBlockStart(nn.Module):
	def __init__(self, module):
		super(ResidualBlockStart, self).__init__()
		self.module = module

	def forward(self, x):
		return self.module(x)

#imapge channekls
# in_channels = 3

# Darknet head (first 20 conv layers + max_pool + fully connected)
# This part is pre-trained on ImageNet and later weights can be loaded into the main YOLO_v1 model before the obj detection training
#
# Note: this could be included in the model.py but since the pre-training of head is typically just 
# done once, I fully seperated Darknet pretraining for better clarity of main code.
class Darknet19(nn.Module):
	
	# in_channels = amount of channels of image (e.g. 3 = RGB, 1 = black and white)
	# split_size (S) = grid size ( split_size x split_size cells)
	# num_boxes (B) = bounding boxes per cell
	# num_classes (C) = amount of classes 
	def __init__(self, classification_head = False , in_channels=3):
		super().__init__()

		#define network elements
		#layers = []
		layers = nn.ModuleList()

		#
		layers.append( nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False) ) #in channels equal to channels in the image
		layers.append( nn.BatchNorm2d(num_features=32) ) #input equal to output size of prior layer
		layers.append( nn.LeakyReLU(0.1) )
		#
		layers.append( nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) )
		#
		layers.append( nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False) )#in channels equal to output of prior
		layers.append( nn.BatchNorm2d(num_features=64) ) #input equal to output size of prior layer
		layers.append( nn.LeakyReLU(0.1) )
		#
		layers.append( nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) )
		#
		layers.append( nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False) ) #in channels equal to output of prior
		layers.append( nn.BatchNorm2d(num_features=128) ) #input equal to output size of prior layer
		layers.append( nn.LeakyReLU(0.1) )
		#
		layers.append( nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False) ) #in channels equal to output of prior
		layers.append( nn.BatchNorm2d(num_features=64) ) #input equal to output size of prior layer
		layers.append( nn.LeakyReLU(0.1) )
		#
		layers.append( nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False) ) #in channels equal to output of prior
		layers.append( nn.BatchNorm2d(num_features=128) ) #input equal to output size of prior layer 
		layers.append( nn.LeakyReLU(0.1) )
		#
		layers.append( nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) )
		#
		layers.append( nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False) ) #in channels equal to output of prior
		layers.append( nn.BatchNorm2d(num_features=256) ) #input equal to output size of prior layer
		layers.append( nn.LeakyReLU(0.1) )
		# 
		layers.append( nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False) ) #in channels equal to output of prior
		layers.append( nn.BatchNorm2d(num_features=128) ) #input equal to output size of prior layer
		layers.append( nn.LeakyReLU(0.1) )
		# 
		layers.append( nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False) ) #in channels equal to output of prior
		layers.append( nn.BatchNorm2d(num_features=256) ) #input equal to output size of prior layer
		layers.append( nn.LeakyReLU(0.1) )
		#
		layers.append( nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) )
		# REPEAT 2x
		for i in range(0,2):
			layers.append( nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False) ) #in channels equal to output of prior
			layers.append( nn.BatchNorm2d(num_features=512) ) #input equal to output size of prior layer
			layers.append( nn.LeakyReLU(0.1) )
			##
			layers.append( nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False) ) #in channels equal to output of prior
			layers.append( nn.BatchNorm2d(num_features=256) ) #input equal to output size of prior layer
			layers.append( nn.LeakyReLU(0.1) )
		# 
		layers_sequence = []
		layers_sequence += [ nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False) ] #in channels equal to output of prior
		layers_sequence += [ nn.BatchNorm2d(num_features=512) ] #input equal to output size of prior layer
		layers_sequence += [ nn.LeakyReLU(0.1) ]

		layers.append( ResidualBlockStart( nn.Sequential(*layers_sequence) ) ) #output of this layer is later used in YOLOv2 architecture
		# layers.append( nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False) ) #in channels equal to output of prior
		# layers.append( nn.BatchNorm2d(num_features=512) ) #input equal to output size of prior layer
		# layers.append( nn.LeakyReLU(0.1) )
		#
		layers.append( nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) )
		# REPEAT 2x
		for i in range(0,2):
			layers.append( nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False) ) #in channels equal to output of prior
			layers.append( nn.BatchNorm2d(num_features=1024) ) #input equal to output size of prior layer
			layers.append( nn.LeakyReLU(0.1) )
			##
			layers.append( nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False) ) #in channels equal to output of prior
			layers.append( nn.BatchNorm2d(num_features=512) ) #input equal to output size of prior layer
			layers.append( nn.LeakyReLU(0.1) )
		# 
		layers.append( nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False) ) #in channels equal to output of prior
		layers.append( nn.BatchNorm2d(num_features=1024) ) #input equal to output size of prior layer
		layers.append( nn.LeakyReLU(0.1) )

		# --- darknet ends here (above pre-trained on ImageNet) ---
		# additional layers to reshape output for ImageNet classification task
		if (classification_head):
			layers.append( nn.Conv2d(in_channels=1024, out_channels=1000, kernel_size=1, stride=1, padding=0, bias=False) ) #in channels equal to output of prior
			layers.append( nn.BatchNorm2d(num_features=1000) ) #input equal to output size of prior layer
			layers.append( nn.AdaptiveAvgPool2d((1, 1)) ) #this produces [batch_size, 1000, 1,1] #[nn.AvgPool2d(kernel_size = 7)]
			layers.append( Reshape() ) #remove "1" dimentions from end  - using reshape instead of squeeze to keep batch dimentions for special case with one sample
			#comment out softmax -> its already included in the CrossEntropyLoss used later during training for classification
			#layers += [nn.Softmax(dim=1)] #dim = 0 is image count dimention, do softmax along the "1000" dimentions to produce class probabilities
		#layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
		#layers += [nn.Linear(1024 * split_size * split_size, 1000)] #final output is 1 number per each class (1000-class image net classifciation)

		
		self.darknet_layers = layers #= nn.Sequential(*layers)
		

	# @param _input image 448x448, tensor size = (x,3,448,448) = (x,N_CHANNEL,WIDTH,HEIGHT) , WHERE: x=num samples, N_CHANNEL = no channels in image (eg RGB =3), WIDTH, HEIGHT = image dimentions
	def forward(self, input_):
		#run the forward pass  (ie. link blocks defined in the init)
		#TODO
		#x = self.darknet_layers(input_)
		x = input_
		for layer in self.darknet_layers:
			x = layer(x)
		return x
		# x = self.darknet(x)
		# return self.fcs(torch.flatten(x, start_dim=1))

		# a1 = self.fc1(input_)
		# h1 = self.relu1(a1)
		# a2 = self.fc2(h1)
		# h2 = self.relu2(a2)
		# a3 = self.out(h2)
		# y = self.out_act(a3)

		# #sequential takes list of layers and chains their input/output together
		# return y


# def test():
# 	model = Darknet19(in_channels=3, classification_head = True)
# 	x = torch.randn((2,3,448,448)) #2 images, each with  3 channels and 448x448 dimentions
#	x = torch.randn((2,3,224,224)) #2 images, each with  3 channels and 224x224 dimentions
# # 	#x = torch.randn((2,3,400,400))
# 	output = model(x)
# 	# print(output)
# 	print(output.shape)
# 	print(torch.sum(output,dim=1))

# test()