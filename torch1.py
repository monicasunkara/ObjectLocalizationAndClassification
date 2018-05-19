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
#testloaders = data.get_test_loader()
valloaders = data.get_val_loader()

resnet18 = models.resnet18(pretrained=True)
modules=list(resnet18.children())[:-1]
resnet18=nn.Sequential(*modules)
for p in resnet18.parameters():
    p.requires_grad = False

D_1, D_2, D_3, D_4, D_5 = 25088, 128, 64, 32, 4

model_reg = torch.nn.Sequential(
    #torch.nn.Linear(25088, 128),
    torch.nn.Linear(512, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 4)
)

model_class = torch.nn.Sequential(
    #torch.nn.Linear(25088, 2408),
    torch.nn.Linear(512, 256),
    torch.nn.BatchNorm1d(256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(256, 1000),
    torch.nn.Softmax()
)

#Use GPU
print("[Using all the available GPUs]")
resnet18 = nn.DataParallel(resnet18.cuda(), device_ids=[0,1])
model_reg = nn.DataParallel(model_reg.cuda(), device_ids=[0,1])
model_class = nn.DataParallel(model_class.cuda(), device_ids=[0,1])


loss_class = torch.nn.CrossEntropyLoss()
loss_reg = torch.nn.MSELoss(size_average=False)

print("Type:")
print(model_reg.parameters())

l_1 = list(filter(lambda p: p.requires_grad, model_reg.parameters()))
l_2 = list(filter(lambda p: p.requires_grad, model_class.parameters()))

optimizer_conv = optim.SGD(l_1+l_2, lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

#train model

since = time.time()

best_acc = 0.0
epochs = 100

for epoch in range(epochs):
	reg_error = 0
	cls_error = 0
	total = 0
	print('Epoch {}/{}'.format(epoch, epochs - 1))
	print('-' * 10)
		
	for phase in ['train']:
		exp_lr_scheduler.step()
		model_reg.train(True)  # Set model to training mode
		model_class.train(True)
		resnet18.train(True)
		running_loss = 0.0
		running_corrects = 0
		sum_loss_cls = 0.0
        	sum_loss_reg = 0.0
		# Iterate over data.
		for data_1  in trainloaders:
			inputs, labels = data_1
			
			# wrap them in Variable
			inputs = Variable(inputs)
			labels_cls = Variable(labels.narrow(1,0,1).cuda()).long()
			labels_reg = Variable(labels.narrow(1, 1, 4).cuda()).float()
			# zero the parameter gradients
			optimizer_conv.zero_grad()
			# forward
			pretrain_out = resnet18(inputs)
		        print(pretrain_out.shape)
			pretrain_out = pretrain_out.view(pretrain_out.size(0), -1)
			outputs_reg = model_reg(pretrain_out)
			outputs_class = model_class(pretrain_out)
			if type(outputs_class) == tuple:
				outputs_class, _ = outputs_class
			_, preds = torch.max(outputs_class.data, 1)
			total+=1
			print(total)
			#Loss
			#print(labels_reg.shape)
			loss_cls = loss_class(outputs_class, labels_cls.squeeze(1))
			loss_regi = loss_reg(outputs_reg, labels_reg)
			#print("loss_reg: " + str(loss_regi.data))
			sum_loss_cls += loss_cls.data[0]
			sum_loss_reg += loss_regi.data[0]
			loss = loss_cls + (0.001*loss_regi)
			if total%100==0:
				print("...." + str(total) + "....")
			# backward + optimize only if in training phase
			if phase == 'train':
				loss.backward()
				optimizer_conv.step()
		print("--Losses--")
		print("Class_Error: " + str(sum_loss_cls))
		print("Reg_Error: " + str(sum_loss_reg))
			#print("total: " + str(total))
	
	predicted = []
	err=0
	te=1
	total_err=0
	class_err=0
	reg_err=0
	total = 0
	for data_2 in valloaders:
        	te=te+1
        	#if te>10:
               	#	break
        	#print(str(te))
        	#predicted = []
        	err=0
        	inputs, labels = data_2
       		inputs = Variable(inputs, volatile=True)
        	resnet18.eval()
        	model_reg.eval()
        	model_class.eval()
       		pretrain_out = resnet18(inputs)
     	   	pretrain_out = pretrain_out.view(pretrain_out.size(0), -1)
        	outputs_reg = model_reg(pretrain_out)
        	outputs_class = model_class(pretrain_out)
        	if type(outputs_class) == tuple:
                	outputs_class, _ = outputs_class
       		_, preds = torch.max(outputs_class, 1)
        	outputs_reg = (outputs_reg.data).cpu().numpy()
        	outputs_reg = outputs_reg[0]
        	#outputs_reg = np.multiply(outputs_reg, std)+mean
		#Calculate IOU
       	 	xA = max(outputs_reg[0], np.float32(labels[1]))
        	yA = max(outputs_reg[1], np.float32(labels[2]))
        	xB = min(outputs_reg[2], np.float32(labels[3]))
        	yB = min(outputs_reg[3], np.float32(labels[4]))
        	interArea = (xB - xA + 1) * (yB - yA + 1)
        	#interArea = interArea[0]
        	boxAArea = (outputs_reg[2] - outputs_reg[0] + 1) * (outputs_reg[3] - outputs_reg[1] + 1)
        	boxBArea = (labels[3] - labels[1] + 1) * (labels[4] - labels[2] + 1)
        	interArea = np.asscalar(interArea)
        	boxAArea = np.asscalar(boxAArea)
        	boxBArea = float(boxBArea)
        	iou = interArea / (boxAArea + boxBArea - interArea)
        	output = [str(float(preds.data)), str(float(labels[0])), str(outputs_reg[0]),str(outputs_reg[1]), str(outputs_reg[2]), str(outputs_reg[3]), str(float(labels[1])), str(float(labels[2])), str(float(labels[3])), str(float(labels[4])), str(iou)]
        	predicted.append(output)
        	#print(str(iou))
        	if iou<0.5:
                	reg_err+=1
                	err=1
		#classification error   
        	if float(preds.data)!=float(labels[0]):
                	class_err+=1
        	err=1
        	if err>0:
                	total_err+=1
        	total+=1
	print("Classification err: " + str(class_err))
	print("Regression err: " + str(reg_err))
	print("total_err: " + str(total_err))
	print("total: " + str(total))
	fileName = "/home/ms3459/DKL/output" + str(epoch)  +  ".csv"
	print(fileName)
	with open(fileName, "wb") as f:
		writer = csv.writer(f)
		#writer.writerow(["ImageId", "PredictionString"])
		writer.writerows(predicted)

#time_elapsed = time.time() - since
#print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
'''
print('---Validation---')
predicted = []
err=0
te=1
total_err=0
class_err=0
reg_err=0
total = 0
for data_2 in valloaders:
        te=te+1
        if te>10000:
        	break
        print(str(te))
	#predicted = []
	err=0
        inputs, labels = data_2
	inputs = Variable(inputs, volatile=True)
        resnet18.eval()
        model_reg.eval()
        model_class.eval()
        pretrain_out = resnet18(inputs)
        pretrain_out = pretrain_out.view(pretrain_out.size(0), -1)
        outputs_reg = model_reg(pretrain_out)
        outputs_class = model_class(pretrain_out)
        if type(outputs_class) == tuple:
                outputs_class, _ = outputs_class
        _, preds = torch.max(outputs_class, 1)
        outputs_reg = (outputs_reg.data).cpu().numpy()
	outputs_reg = outputs_reg[0]
	outputs_reg = np.multiply(outputs_reg, std)+mean
	#Calculate IOU
	xA = max(outputs_reg[0], np.float32(labels[1]))
	yA = max(outputs_reg[1], np.float32(labels[2]))
	xB = min(outputs_reg[2], np.float32(labels[3]))
	yB = min(outputs_reg[3], np.float32(labels[4]))
	interArea = (xB - xA + 1) * (yB - yA + 1)
	#interArea = interArea[0]
	boxAArea = (outputs_reg[2] - outputs_reg[0] + 1) * (outputs_reg[3] - outputs_reg[1] + 1)
	boxBArea = (labels[3] - labels[1] + 1) * (labels[4] - labels[2] + 1)
	interArea = np.asscalar(interArea)
	boxAArea = np.asscalar(boxAArea)
	boxBArea = float(boxBArea)
	iou = interArea / (boxAArea + boxBArea - interArea)
	output = [str(float(preds.data)), str(float(labels[0])), str(outputs_reg[0]),str(outputs_reg[1]), str(outputs_reg[2]), str(outputs_reg[3]), str(float(labels[1])), str(float(labels[2])), str(float(labels[3])), str(float(labels[4])), str(iou)]
	predicted.append(output)
	print(str(iou))
	if iou<0.5:
		reg_err+=1
		err=1
	#classification error	
	if float(preds.data)!=float(labels[0]):
		class_err+=1
	err=1
	if err>0:
		total_err+=1
	total+=1
	print("Classification err: " + str(class_err))
	print("Regression err: " + str(reg_err))
	print("total_err: " + str(total_err))
	print("total: " + str(total))
	fileName = "/home/ms3459/DKL/output.csv"
	with open("/home/ms3459/DKL/output.csv", "wb") as f:
        writer = csv.writer(f)
        #writer.writerow(["ImageId", "PredictionString"])
        writer.writerows(predicted)
'''
'''
#Test

print('---Testing---')
predicted = []
count = '00000000'
te=1
for data_2 in testloaders:
	#te=te+1
	#if te>15:
	#	break
	#predicted = []
	inputs, labels = data_2
	inputs = Variable(inputs, volatile=True)
	resnet18.eval()
	model_reg.eval()
	model_class.eval()
	pretrain_out = resnet18(inputs)
	print(pretrain_out.size())
	pretrain_out = pretrain_out.view(pretrain_out.size(0), -1)
	print(pretrain_out.size())
	outputs_reg = model_reg(pretrain_out)
	#print(outputs_reg.size())
	outputs_class = model_class(pretrain_out)
	#print(outputs_class.size())
	if type(outputs_class) == tuple:
        	outputs_class, _ = outputs_class
	#print(outputs_class)
	#outputs_class = torch.stack([nn.Softmax(dim=0)(i) for i in outputs_class])
        #print(outputs_class)
	#outputs_class = outputs_class.mean(0)
        _, preds = torch.max(outputs_class, 1)
	outputs_reg = outputs_reg.data
	#count = '%08d' % (int(count) + 1)
	#print(outputs_class.size())
	#print(len(preds.data))
	#print(outputs_reg.size())
	i=0
	while i<len(preds.data):
		#mini, norm = metaData.getVals()
		count = '%08d' % (int(count) + 1)
		print(mean[0])
		print(std[0]) 
        	output_1 = ['ILSVRC2012_test_' + str(count)  , str(preds.data[i]) + ' ' + str(outputs_reg[i][0]*std[0]+mean[0]) + ' ' + str(outputs_reg[i][1]*std[1]+mean[1]) + ' ' + str(outputs_reg[i][2]*std[2]+mean[2]) +  ' ' + str(outputs_reg[i][3]*std[3]+mean[3])]
		predicted.append(output_1)	
		#print(count)
		i=i+1
#print(len(predicted))
#print(predicted)

with open("/home/ms3459/DKL/output.csv", "wb") as f:
	writer = csv.writer(f)
	writer.writerow(["ImageId", "PredictionString"])
    	writer.writerows(predicted)
'''
