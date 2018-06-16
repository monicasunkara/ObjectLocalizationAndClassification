import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import metaData
import torchvision.datasets as dsets
import testData
import valData

def get_train_loader():
	input_shape = 224
	batch_size = 32
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	scale = 360
	input_shape = 224
	use_parallel = True
	use_gpu = True
	epochs = 100
	data_transforms = transforms.Compose([
        transforms.Resize(scale),
        transforms.RandomResizedCrop(input_shape),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
	dataset = metaData.MetaDataset(transform=data_transforms)
        data_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True, num_workers=4)
	mean, std = dataset.getVals
	return data_loader, mean, std

def get_test_loader():
        input_shape = 224
        batch_size = 2	
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        scale = 360
        input_shape = 224
        use_parallel = True
        use_gpu = True
        epochs = 100
        data_transforms = transforms.Compose([
        transforms.Resize(scale),
        transforms.RandomResizedCrop(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
        image_datasets = testData.TestDataset(transform=data_transforms)
	data_loader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size,shuffle=True, num_workers=4)
	return data_loader

def get_val_loader():
	input_shape = 224
        batch_size = 256
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        scale = 360
        input_shape = 224
        use_parallel = True
        use_gpu = True
        epochs = 100
        data_transforms = transforms.Compose([
        transforms.Resize(scale),
        transforms.CenterCrop(224),
	transforms.RandomResizedCrop(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
        image_datasets = valData.ValDataset(transform=data_transforms)
        data_loader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size,shuffle=True, num_workers=4)
        return data_loader

