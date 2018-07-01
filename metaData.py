import torch
from PIL import Image
from random import shuffle
import json
import numpy as np
import torchvision.transforms as transforms
import os
import voc_annotations_parser as voc 
import cPickle as  pickle

class MetaDataset:

	def __init__(self, transform = transforms.ToTensor()):
		rootdir = '/home/ms3459/DKLData/ILSVRC'
		IMAGE_SET_PATH = os.path.join(rootdir, 'ImageSets/CLS-LOC/train_loc_small.txt')
		ANNOTATIONS_PATH = os.path.join(rootdir, 'Annotations/CLS-LOC/train')
		# Image path
		IMAGE_PATH = os.path.join(rootdir, 'Data/CLS-LOC/traincopy')
		self.dirlist = [ item for item in os.listdir(IMAGE_PATH) if os.path.isdir(os.path.join(IMAGE_PATH, item)) ]
		#parser = voc.VocAnnotationsParser(IMAGE_PATH, IMAGE_SET_PATH, ANNOTATIONS_PATH)
		#with open('/home/ms3459/DKL/Alllabels.pkl', 'wb') as f:
		#	mylist = parser.annotation_line_list
		#	pickle.dump(mylist, f)
		with open('/home/ms3459/DKL/Alllabelssmall.pkl', 'rb') as f:
			self.annon_list = pickle.load(f)
		shuffle(self.annon_list)
		a_list = [[d['xmin'], d['ymin'], d['xmax'], d['ymax']] for d in self.annon_list]
		annon_np = np.array(a_list).astype(np.float64)
                self.mean = annon_np.mean(axis=0)
		self.std = annon_np.std(axis=0)
		annon_np = annon_np - self.mean
		annon_np = annon_np/self.std
		self.norm_list = annon_np
		self.transform = transform
		self.rootdir = rootdir
                
	def __getitem__(self,i):
                annotation = self.annon_list[i]
		count = 1
        	image_path = annotation['img_full_path']
		if os.path.exists(image_path):
        		img = Image.open(image_path).convert('RGB')
			img = self.transform(img)
			target = np.array([self.dirlist.index(annotation['foldername']), self.norm_list[i][0], self.norm_list[i][1], self.norm_list[i][2], self.norm_list[i][3]])
			return img, target
		print("No path found: " + str(image_path))
		nums = [1, 2, 3, 4, 5]
		return torch.ones([3, 224, 224]), np.asarray(nums)

	def __len__(self):
        	return len(self.annon_list)
	@property
	def getVals(self):
		return self.mean, self.std
