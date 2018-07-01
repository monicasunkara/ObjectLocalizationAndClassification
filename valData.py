import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
import voc_annotations_parser as voc
import pickle

class ValDataset:

        def __init__(self, transform = transforms.ToTensor()):
                rootdir = '/home/ms3459/DKLData/ILSVRC'
                IMAGE_SET_PATH = os.path.join(rootdir, 'ImageSets/CLS-LOC/val1.txt')
                # path to the voc annotation folder
                ANNOTATIONS_PATH = os.path.join(rootdir, 'Annotations/CLS-LOC/val')
                # Image path
                IMAGE_PATH = os.path.join(rootdir, 'Data/CLS-LOC/val')
                self.dirlist = [ item for item in os.listdir(os.path.join(rootdir, 'Data/CLS-LOC/train')) if os.path.isdir(os.path.join(rootdir, 'Data/CLS-LOC/train', item)) ]
                with open('/home/ms3459/DKL/Alllabelsval.pkl', 'rb') as f:
                        self.annon_list = pickle.load(f)
                self.transform = transform
		self.rootdir = rootdir

        def __getitem__(self,i):
                annotation = self.annon_list[i]
		image_path = annotation['img_full_path']
		if os.path.exists(image_path):
                        img = Image.open(image_path).convert('RGB')
                        img = self.transform(img)
                        target = np.array([self.dirlist.index(annotation['foldername']), annotation['xmin'], annotation['ymin'], annotation['xmax'], annotation['ymax']])
			return img, target
                return [], []

        def __len__(self):
                return len(self.annon_list)
