import torch
from torch import Tensor
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import time
import os
import metaData
import data
import argparse
from torch.autograd import Variable
import csv

#Data loader
from torchvision import transforms, datasets, utils

#true values
trainloaders, mean, std = data.get_train_loader()
print("Training data loaded")

valloaders = data.get_val_loader()
print("Validation data loaded")

resnet18 = models.resnet18(pretrained=True)
modules=list(resnet18.children())[:-1]
resnet18=nn.Sequential(*modules)
for param in resnet18.parameters():
    param.requires_grad = False

model_reg = torch.nn.Sequential(
    torch.nn.Linear(512, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(128, 32),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(32, 4)
)

model_class = torch.nn.Sequential(
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(128, 1000)
)

#Use GPU
print("[Using all the available GPUs]")
resnet18 = nn.DataParallel(resnet18.cuda(), device_ids=[0,1])
model_reg = nn.DataParallel(model_reg.cuda(), device_ids=[0,1])
print("Parallel")
model_class = nn.DataParallel(model_class.cuda(), device_ids=[0,1])
print("Model parallel set")
loss_class = torch.nn.CrossEntropyLoss().cuda()
loss_reg = torch.nn.MSELoss().cuda()
print("Losses")
l_1 = list(filter(lambda p: p.requires_grad, model_reg.parameters()))
l_2 = list(filter(lambda p: p.requires_grad, model_class.parameters()))

optimizer_conv = optim.SGD(l_2+l_1, lr=0.01, momentum=0.9)

#train model

since = time.time()

best_acc = 0.0

epochs = 50

exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_conv, milestones=[int(0.5 * epochs), int(0.75 * epochs)], gamma=0.1)

for epoch in range(epochs):
	reg_error = 0
	cls_error = 0
	total = 0
	print('Epoch {}/{}'.format(epoch, epochs - 1))
	print('-' * 10)
	for phase in ['train']:
		exp_lr_scheduler.step()	
		model_reg.train(True) 
		model_class.train(True)
		resnet18.train(True)
		running_loss = 0.0
		running_corrects = 0
		sum_loss_cls = 0.0
		sum_loss_reg = 0.0
		
		# Iterate over data.
		for data in trainloaders:
			inputs, labels = data
			# wrap them in Variable
			inputs = Variable(inputs)
			labels_cls = Variable(labels.select(1,0).cuda()).long()
			labels_reg = Variable(labels.narrow(1,1,4).cuda()).float()
		
			# zero the parameter gradients
			optimizer_conv.zero_grad()
			# forward
			pretrain_out = resnet18(inputs)
			pretrain_out = pretrain_out.view(pretrain_out.size(0),-1)
			outputs_reg = model_reg(pretrain_out)
			outputs_class = model_class(pretrain_out)
			if type(outputs_class) == tuple:
				outputs_class, _ = outputs_class
			_, preds = torch.max(outputs_class.data, 1)
			total+=inputs.shape[0]

			#Loss
			labels = labels.numpy()
			loss_cls = loss_class(outputs_class, labels_cls)
			cls_error += (preds!=labels_cls.data).sum()
			loss_regi = loss_reg(outputs_reg, labels_reg)
			outputs_reg = (outputs_reg.data).cpu().numpy()
			
			#IOU
			xA = np.maximum(outputs_reg[:,0], np.float32(labels[:,1]))
                	yA = np.maximum(outputs_reg[:,1], np.float32(labels[:,2]))
                	xB = np.maximum(outputs_reg[:,2], np.float32(labels[:,3]))
                	yB = np.maximum(outputs_reg[:,3], np.float32(labels[:,4]))
                	interArea = (xB - xA + 1) * (yB - yA + 1)
                	boxAArea = (outputs_reg[:,2] - outputs_reg[:,0] + 1) * (outputs_reg[:,3] - outputs_reg[:,1] + 1)
                	boxBArea = (labels[:,3] - labels[:,1] + 1) * (labels[:,4] - labels[:,2] + 1)
                	iou = interArea / (boxAArea + boxBArea - interArea)
                
                	reg_error+=np.where(iou<0.5)[0].shape[0]
			sum_loss_cls += loss_cls.data[0]
			sum_loss_reg += loss_regi.data[0]
			
			loss = loss_regi + loss_cls
			loss.backward()
			optimizer_conv.step()
		print("Class_error_training: " + str(cls_error))
		print("Reg_Error: " + str(reg_error))
		print("Total: " + str(total))
	
	class_err=0
	reg_err=0
	total = 0
	pred_set = set()
	for data_2 in valloaders:
		
		inputs, labels = data_2
		labels_cls = Variable(labels.select(1,0).cuda()).long()
		labels_reg = Variable(labels.narrow(1,1,4).cuda()).float()
		inputs = Variable(inputs)
		resnet18.eval()
		model_reg.eval()
		model_class.eval()
		pretrain_out = resnet18(inputs)
		pretrain_out=pretrain_out.view(pretrain_out.size(0),-1)
		outputs_reg = model_reg(pretrain_out)
		outputs_class = model_class(pretrain_out)
		if type(outputs_class) == tuple:
			outputs_class, _ = outputs_class
		_, preds = torch.max(outputs_class, 1)
		
		outputs_reg = (outputs_reg.data).cpu().numpy()
		labels = labels.numpy()
		outputs_reg = np.multiply(outputs_reg, std)+mean
		
		#Calculate IOU
		
		xA = np.maximum(outputs_reg[:,0], np.float32(labels[:,1]))
		yA = np.maximum(outputs_reg[:,1], np.float32(labels[:,2]))
		xB = np.maximum(outputs_reg[:,2], np.float32(labels[:,3]))
		yB = np.maximum(outputs_reg[:,3], np.float32(labels[:,4]))
		#print(xB, xA, yB, yA)
		interArea = (xB - xA + 1) * (yB - yA + 1)
		#print(interArea.shape)
		#interArea = interArea[0]
		boxAArea = (outputs_reg[:,2] - outputs_reg[:,0] + 1) * (outputs_reg[:,3] - outputs_reg[:,1] + 1)
		boxBArea = (labels[:,3] - labels[:,1] + 1) * (labels[:,4] - labels[:,2] + 1)
		iou = interArea / (boxAArea + boxBArea - interArea)	
		reg_err+=np.where(iou<0.5)[0].shape[0]
		class_err += (preds.data!=labels_cls.data).sum()
		total+=len(preds)
	print("Classification err: " + str(class_err))
	print("Regression err: " + str(reg_err))
	print("total: " + str(total))
