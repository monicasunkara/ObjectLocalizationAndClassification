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
print("Training loaded...")
#testloaders = data.get_test_loader()
valloaders = data.get_val_loader()

resnet18 = models.resnet18(pretrained=True)
modules=list(resnet18.children())[:-1]
resnet18=nn.Sequential(*modules)
for param in resnet18.parameters():
    param.requires_grad = False

#num_ftrs = resnet18.fc.in_features
#resnet18.fc = nn.Linear(num_ftrs, 1000)

D_1, D_2, D_3, D_4, D_5 = 25088, 128, 64, 32, 4

model_reg = torch.nn.Sequential(
    #torch.nn.Linear(25088, 128),
    torch.nn.Linear(512, 32),
     torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
   #torch.nn.Linear(128, 32),
    #torch.nn.ReLU(),
    torch.nn.Linear(32, 4)
)

model_class = torch.nn.Sequential(
    #torch.nn.Linear(25088, 2408),
    #torch.nn.Linear(512, 256),
    #torch.nn.BatchNorm1d(256),
    #torch.nn.ReLU(),
    #torch.nn.Dropout(0.4),
    torch.nn.Linear(512, 1000),
    #torch.nn.ReLU(),
    #torch.nn.Linear(256, 1000)
)

#Use GPU
print("[Using all the available GPUs]")
resnet18 = nn.DataParallel(resnet18).cuda()#, device_ids=[0,1])
model_reg = nn.DataParallel(model_reg).cuda()#, device_ids=[0,1])
print("Parallel")
model_class = nn.DataParallel(model_class).cuda()#, device_ids=[0,1])
print("Model parallel set")
loss_class = torch.nn.CrossEntropyLoss().cuda()
loss_reg = torch.nn.MSELoss().cuda()
print("Losses")
l_1 = list(filter(lambda p: p.requires_grad, model_reg.parameters()))
l_2 = list(filter(lambda p: p.requires_grad, model_class.parameters()))
print("param")
optimizer_conv = optim.SGD(l_2+l_1, 0.01)
print("param set")
#train model

since = time.time()

best_acc = 0.0
epochs = 600000

#exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_conv, milestones=[int(0.5 * epochs), int(0.75 * epochs)], gamma=0.1)

for epoch in range(epochs):
	reg_error = 0
	cls_error = 0
	train_class_set = set()
	total = 0
	print('Epoch {}/{}'.format(epoch, epochs - 1))
	print('-' * 10)

	for phase in ['train']:
		#exp_lr_scheduler.step()
		model_reg.train(True)  # Set model to training mode
		model_class.train(True)
		resnet18.train(True)
		running_loss = 0.0
		running_corrects = 0
		sum_loss_cls = 0.0
		sum_loss_reg = 0.0
		toti = 0
		# Iterate over data.
		for data_1  in trainloaders:
			#toti+=1
			#if toti==2:
			#	break
			inputs, labels = data_1	
			# wrap them in Variable
			inputs = Variable(inputs)
			labels_cls = Variable(labels.select(1,0).cuda()).long()
			labels_reg = Variable(labels.narrow(1,1,4).cuda()).float()
			#print(labels_reg)
			# zero the parameter gradients
			optimizer_conv.zero_grad()
			# forward
			pretrain_out = resnet18(inputs)
			#print(pretrain_out.shape)
			pretrain_out = pretrain_out.view(pretrain_out.size(0),-1)
			#pretrain_out = pretrain_out.detach()
			#pretrain_out.volatile = False
			outputs_reg = model_reg(pretrain_out)
			outputs_class = model_class(pretrain_out)
			if type(outputs_class) == tuple:
				outputs_class, _ = outputs_class
			_, preds = torch.max(outputs_class.data, 1)
			total+=inputs.shape[0]

			for class_label in labels_cls.data.tolist():
				train_class_set.add(class_label)
			#Loss
			#print(labels_reg.shape)
			loss_cls = loss_class(outputs_class, labels_cls)
			cls_error += (preds!=labels_cls.data).sum()
			loss_regi = loss_reg(outputs_reg, labels_reg)
			#print("loss_reg: " + str(loss_regi.data))
			#print("Loss_cls: " + str(loss_cls.data))
			sum_loss_cls += loss_cls.data[0]
			sum_loss_reg += loss_regi.data[0]
			loss = loss_cls + 2*loss_regi
			loss.backward()
			optimizer_conv.step()
			#print(train_class_set)
		print("--Losses--")
		print("Class_loss: " + str(sum_loss_cls))
		print("Class_error_training: " + str(cls_error))
		#print("Reg_Error: " + str(sum_loss_reg))

	predicted = []
	err=0
	te=1
	class_err=0
	reg_err=0
	total = 0
	pred_set = set()
	for data_2 in valloaders:
		
		te=te+1
		#if te>10:
		#	break
		#print(str(te))
		#predicted = []
		err=0
		inputs, labels = data_2
		labels_cls = Variable(labels.select(1,0).cuda()).long()
		labels_reg = Variable(labels.narrow(1,1,4).cuda()).float()
		inputs = Variable(inputs)
		resnet18.eval()
		model_reg.eval()
		model_class.eval()
		pretrain_out = resnet18(inputs)
		#pretrain_out=pretrain_out.view(pretrain_out.size(0),-1)
		#IO
		pretrain_out=pretrain_out.view(pretrain_out.size(0),-1) 
		#pretrain_out = pretrain_out.detach()
                #pretrain_out.volatile = False

		###
		outputs_reg = model_reg(pretrain_out)
		outputs_class = model_class(pretrain_out)
		if type(outputs_class) == tuple:
			outputs_class, _ = outputs_class
		_, preds = torch.max(outputs_class, 1)
		#pred_set.add(preds.data[0])
		#print(pred_set)
		outputs_reg = (outputs_reg.data).cpu().numpy()
		labels = labels.numpy()
		#outputs_reg = outputs_reg[0]
		outputs_reg = np.multiply(outputs_reg, std)+mean
		
		#Calculate IOU
		#print(labels)
		#print(outputs_reg[0])
		#print(outputs_reg[:,0])
		#print(labels[:,1].shape)
		#print(outputs_reg[:,1].shape)
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
		#print(boxBArea.shape, boxAArea.shape)
		#interArea = np.asscalar(interArea)
		#boxAArea = np.asscalar(boxAArea)
		#boxBArea = float(boxBArea)
		iou = interArea / (boxAArea + boxBArea - interArea)
		#print("IOU:")
		#print(iou.shape)
		#print(iou)
		#output = [str(float(preds.data)), str(float(labels[0])), str(outputs_reg[0]),str(outputs_reg[1]), str(outputs_reg[2]), str(outputs_reg[3]), str(float(labels[1])), str(float(labels[2])), str(float(labels[3])), str(float(labels[4])), str(iou)]
		#predicted.append(output)
		#print(str(iou))
		reg_err+=np.where(iou<0.5)[0].shape[0]
		#print(np.where(iou<0.5))
		#print(np.where(iou<0.5)[0].shape[0])
		#if iou<0.5:
		#	reg_err+=1
		#	err=1
		
		#output = [str(float(preds.data))]
		#predicted.append(output)
		#classification error   
		#print(type(preds.data), type(labels_cls.data))
		#if float(preds.data)!=float(labels_cls.data):
		#print(type(preds.data), type(labels_cls.data))
		class_err += (preds.data!=labels_cls.data).sum()
		#	class_err+=1
		#	print(float(preds.data), float(labels_cls.data))
		#err=1
		#if err>0:
			#total_err+=1
		total+=len(preds)
	print("Classification err: " + str(class_err))
	print("Regression err: " + str(reg_err))
	#print("total_err: " + str(total_err))
	print("total: " + str(total))
	#fileName = "/home/ms3459/DKL/nclsoutput" + str(epoch)  +  ".csv"
	#print(fileName)
	#with open(fileName, "wb") as f:
	#	writer = csv.writer(f)
	#	writer.writerow(["ImageId", "PredictionString"])
	#	writer.writerows(predicted)

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
