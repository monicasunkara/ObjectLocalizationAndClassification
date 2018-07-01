from helpers import get_file_lines
from xml.etree import ElementTree
import pandas as pd
import os

class VocAnnotationsParser(object):

    def __init__(self, voc_image_path, voc_image_set_path, voc_annon_path):
        self.current_work_dir = os.getcwd()
        self.voc_image_path = voc_image_path
        self.voc_image_set_path = voc_image_set_path
        self.voc_annon_path = voc_annon_path
        self._annotation_line_list = [] # a list of annotations, each annontation is dict
        self._parse_from_voc() # parse all the data

    @property
    def annotation_line_list(self):
        return self._annotation_line_list

    def get_annotation_dataframe(self):
        """
        Returns a dataframe with the parsed pascal voc data. When an image has several bbox annotations, the resulting dataframe\
        has a line for each.

        """
        return pd.DataFrame(self.annotation_line_list)

    def _parse_from_voc(self):
        # get all the files names from the voc_imageset_text_path
	filenames_list = get_file_lines(self.voc_image_set_path)
        #for each filename from the image set we need to get the annotations
	for filename in filenames_list:
            # get the path of the annotation file
            annotation_file = self._get_img_detection_filepath(self.voc_annon_path, filename.partition(' ')[0])
	    # tree of the xml
            tree = ElementTree.parse(annotation_file)
            # get the root element
            root_node = tree.getroot()
            # get file name
            img_filename = root_node.find('filename').text
            img_full_path = self._get_img_filepath(filename.partition(' ')[0])
            # get the size of the image from the annotation xml file
            width, height = self._get_img_size(root_node)

            #get the the list of all object trees from the annotation xml
            object_tree_list = root_node.findall('object')
	    if len(object_tree_list)>1:
		continue
            #for each object tree
            for object_annotation in object_tree_list:
                # create a dictionary with all the information
                # {img,img_full_path,width,height,class_name,xmin,ymin,xmax,ymax}
                row_dictionary = {}

                class_name = self._get_annotation_classname(object_annotation)
                img_foldername = object_annotation.find('name').text
		obj_bbox = object_annotation.find('bndbox')
                xmin, ymin, xmax, ymax = self._get_annotation_bbox(obj_bbox)

                # now that we have all the information from an annotation bbox
                # create a dict to be inserted in the final result
                row_dictionary.update({'filename': img_filename,
				       'foldername': img_foldername,
                                       'img_full_path': img_full_path,
                                       'width': width,
                                       'height': height,
                                       'class_name': class_name,
                                       'xmin': xmin,
                                       'ymin': ymin,
                                       'xmax': xmax,
                                       'ymax': ymax})
		self._annotation_line_list.append(row_dictionary)

    def _get_img_detection_filepath(self, voc_annotations_path, img_name):
        return os.path.join(voc_annotations_path, img_name + '.xml')

    def _get_img_filepath(self, image):
        return os.path.join(self.voc_image_path, image + '.JPEG')
        
    def _get_img_size(self, root_node):
        size_tree = root_node.find('size')
        width = float(size_tree.find('width').text)
        height = float(size_tree.find('height').text)
        return (width, height)

    def _get_annotation_classname(self, object_annotation):
        return object_annotation.find('name').text

    def _get_annotation_bbox(self, bbox_node):
        xmin = int(round(float(bbox_node.find('xmin').text)))
        ymin = int(round(float(bbox_node.find('ymin').text)))
        xmax = int(round(float(bbox_node.find('xmax').text)))
        ymax = int(round(float(bbox_node.find('ymax').text)))
        return (xmin, ymin, xmax, ymax)
