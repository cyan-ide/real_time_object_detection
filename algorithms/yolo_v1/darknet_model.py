import torch
import torch.nn as nn
import torch.optim as optim


class Squeeze(nn.Module):
	def __init__(self):
		super(Squeeze, self).__init__()

	def forward(self, x):
		return x.squeeze()

# imapge channels
# in_channels = 3

# Darknet head (first 20 conv layers + max_pool + fully connected)
# This part is pre-trained on ImageNet and later weights can be loaded into the main YOLO_v1 model before the obj detection training
#
# Note: this could be included in the model.py but since the pre-training of head is typically just 
# done once, I fully seperated Darknet pretraining for better clarity of main code.
class Darknet(nn.Module):
	
	# in_channels = amount of channels of image (e.g. 3 = RGB, 1 = black and white)
	# split_size (S) = grid size ( split_size x split_size cells)
	# num_boxes (B) = bounding boxes per cell
	# num_classes (C) = amount of classes 
	def __init__(self, classification_head = False , in_channels=3):
		super().__init__()

		#define network elements
		layers = []

		#
		layers += [nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)]  #in channels equal to channels in the image
		layers += [nn.BatchNorm2d(num_features=64)] #input equal to output size of prior layer
		layers += [nn.LeakyReLU(0.1)]
		#
		layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
		#
		layers += [nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1, bias=False)] #in channels equal to output of prior
		layers += [nn.BatchNorm2d(num_features=192)] #input equal to output size of prior layer
		layers += [nn.LeakyReLU(0.1)]
		#
		layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
		#
		layers += [nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)] #in channels equal to output of prior
		layers += [nn.BatchNorm2d(num_features=128)] #input equal to output size of prior layer
		layers += [nn.LeakyReLU(0.1)]
		#
		layers += [nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)] #in channels equal to output of prior
		layers += [nn.BatchNorm2d(num_features=256)] #input equal to output size of prior layer
		layers += [nn.LeakyReLU(0.1)]
		#
		layers += [nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)] #in channels equal to output of prior
		layers += [nn.BatchNorm2d(num_features=256)] #input equal to output size of prior layer
		layers += [nn.LeakyReLU(0.1)]
		#
		layers += [nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)] #in channels equal to output of prior
		layers += [nn.BatchNorm2d(num_features=512)] #input equal to output size of prior layer
		layers += [nn.LeakyReLU(0.1)]
		# 
		layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
		# REPEAT 4x
		for i in range(0,4):
			layers += [nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)] #in channels equal to output of prior
			layers += [nn.BatchNorm2d(num_features=256)] #input equal to output size of prior layer
			layers += [nn.LeakyReLU(0.1)]
			##
			layers += [nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)] #in channels equal to output of prior
			layers += [nn.BatchNorm2d(num_features=512)] #input equal to output size of prior layer
			layers += [nn.LeakyReLU(0.1)]
		# 
		layers += [nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)] #in channels equal to output of prior
		layers += [nn.BatchNorm2d(num_features=512)] #input equal to output size of prior layer
		layers += [nn.LeakyReLU(0.1)]
		# 
		layers += [nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)] #in channels equal to output of prior
		layers += [nn.BatchNorm2d(num_features=1024)] #input equal to output size of prior layer
		layers += [nn.LeakyReLU(0.1)]
		# 
		layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
		# REPEAT 2x
		for i in range(0,2):
			layers += [nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)] #in channels equal to output of prior
			layers += [nn.BatchNorm2d(num_features=512)] #input equal to output size of prior layer
			layers += [nn.LeakyReLU(0.1)]
			##
			layers += [nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)] #in channels equal to output of prior
			layers += [nn.BatchNorm2d(num_features=1024)] #input equal to output size of prior layer
			layers += [nn.LeakyReLU(0.1)]
		# --- darknet ends here (above pre-trained on ImageNet) ---
		# additional layers to reshape output for ImageNet classification task
		if (classification_head):
			layers += [nn.AvgPool2d(kernel_size = 7)]
			layers += [Squeeze()] #torch.Squeeze()
			layers += [nn.Linear(1024, 1000)] #final output is 1 number per each class (1000-class image net classifciation)
		#layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
		#layers += [nn.Linear(1024 * split_size * split_size, 1000)] #final output is 1 number per each class (1000-class image net classifciation)

		
		self.darknet_layers = nn.Sequential(*layers)
		

	# @param _input image 448x448, tensor size = (x,3,448,448) = (x,N_CHANNEL,WIDTH,HEIGHT) , WHERE: x=num samples, N_CHANNEL = no channels in image (eg RGB =3), WIDTH, HEIGHT = image dimentions
	def forward(self, input_):
		#run the forward pass  (ie. link blocks defined in the init)
		x = self.darknet_layers(input_)
		return x


# def test():
# 	model = YOLO_v1(in_channels=3, split_size=7, num_boxes=2, num_classes=20)
# 	x = torch.randn((2,3,448,448))
# 	#x = torch.randn((2,3,400,400))
# 	print(model(x).shape)

# test()