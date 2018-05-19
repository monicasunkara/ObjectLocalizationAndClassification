import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
import voc_annotations_parser as voc

class ValDataset:

        def __init__(self, transform = transforms.ToTensor()):
                rootdir = '/home/ms3459/DKLData/ILSVRC'
                IMAGE_SET_PATH = os.path.join(rootdir, 'ImageSets/CLS-LOC/val3.txt')
                # path to the voc annotation folder

                ANNOTATIONS_PATH = os.path.join(rootdir, 'Annotations/CLS-LOC/val')
                # Image path
                IMAGE_PATH = os.path.join(rootdir, 'Data/CLS-LOC/val')
                self.dirlist = [ item for item in os.listdir(os.path.join(rootdir, 'Data/CLS-LOC/train')) if os.path.isdir(os.path.join(rootdir, 'Data/CLS-LOC/train', item)) ]
                #print(self.dirlist)
		parser = voc.VocAnnotationsParser(IMAGE_PATH, IMAGE_SET_PATH, ANNOTATIONS_PATH)
                self.annon_list = parser.annotation_line_list
                '''
		#annon_np = np.array(self.annon_list)
                #print(type(annon_np.min(axis=0)))
                #print(annon_np.min(axis=0))
                #a_list = [[d['xmin'], d['ymin'], d['xmax'], d['ymax']] for d in self.annon_list]
                #print(a_list)
                annon_np = np.array(a_list).astype(np.float64)
                #annon_np = annon_arr[:, [6,7,8,9]]
                print('############################################################################')
                #print(annon_np[0])
                self.mean = annon_np.mean(axis=0)
                self.std = annon_np.std(axis=0)
                print(self.mean)
                print(self.std)
                annon_np = annon_np - self.mean
                annon_np = annon_np/self.std
                self.norm_list = annon_np
                #self.annon_df = parser.get_annotation_dataframe_compact()
                '''
		self.transform = transform
                self.rootdir = rootdir

        def __getitem__(self,i):
                annotation = self.annon_list[i]
                #print(annotation)
		image_path = annotation['img_full_path']
                #print(image_path)
		if os.path.exists(image_path):
                        img = Image.open(image_path).convert('RGB')
                        img = self.transform(img)
                        #print("Helloooo")
			#print(image_path)
			#print(type(img))
                        #printt(img)
                        target = [self.dirlist.index(annotation['foldername']), annotation['xmin'], annotation['ymin'], annotation['xmax'], annotation['ymax']]
			#print(img)
			#print(target)
                        return img, target
                return [], []

        def __len__(self):
                return len(self.annon_list)
        
	#@property
        #def getVals(self):
        #        return self.mean, self.std
