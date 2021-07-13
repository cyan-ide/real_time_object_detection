import torch
import torch.nn as nn
import torch.optim as optim

#imapge channekls
# in_channels = 3

# 2. network architecture
class YOLO_v1(nn.Module):
	
	# darknet_layers - darknet neural network layers
	# in_channels = amount of channels of image (e.g. 3 = RGB, 1 = black and white)
	# split_size (S) = grid size ( split_size x split_size cells)
	# num_boxes (B) = bounding boxes per cell
	# num_classes (C) = amount of classes 
	def __init__(self, darknet_layers, in_channels=3, split_size=7, num_boxes=2, num_classes=20):
		super().__init__()

		#define network elements
		layers = []
		layers += darknet_layers
		# --- darknet ends here (above pre-trained on ImageNet) ---
		layers += [nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)] #in channels equal to output of prior
		layers += [nn.BatchNorm2d(num_features=1024)] #input equal to output size of prior layer
		layers += [nn.LeakyReLU(0.1)]
		# 
		layers += [nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1, bias=False)] #in channels equal to output of prior
		layers += [nn.BatchNorm2d(num_features=1024)] #input equal to output size of prior layer
		layers += [nn.LeakyReLU(0.1)]
		# 
		layers += [nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)] #in channels equal to output of prior
		layers += [nn.BatchNorm2d(num_features=1024)] #input equal to output size of prior layer
		layers += [nn.LeakyReLU(0.1)]
		# 
		layers += [nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)] #in channels equal to output of prior
		layers += [nn.BatchNorm2d(num_features=1024)] #input equal to output size of prior layer
		layers += [nn.LeakyReLU(0.1)]
		#
		layers += [nn.Flatten()]
		#
		layers += [nn.Linear(1024 * split_size * split_size, 496)]
		#
		layers += [nn.LeakyReLU(0.1)]
		#
		layers += [nn.Linear(496, split_size * split_size * (num_classes + num_boxes * 5))] # "5" = width,height, xCenter,yCenter, obj_probability
		
		self.yolo_v1 = nn.Sequential(*layers)
		
	# @param _input image 448x448, tensor size = (x,3,448,448) = (x,N_CHANNEL,WIDTH,HEIGHT) , WHERE: x=num samples, N_CHANNEL = no channels in image (eg RGB =3), WIDTH, HEIGHT = image dimentions
	def forward(self, input_):
		#run the forward pass  (ie. link blocks defined in the init)
		#TODO
		x = self.yolo_v1(input_)
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
# 	model = YOLO_v1(in_channels=3, split_size=7, num_boxes=2, num_classes=20)
# 	x = torch.randn((2,3,448,448))
# 	#x = torch.randn((2,3,400,400))
# 	print(model(x).shape)

# test()