from easyocr.detection import get_detector, get_textbox, test_net
from easyocr.imgproc import loadImage, resize_aspect_ratio, normalizeMeanVariance
from easyocr.craft_utils import getDetBoxes, adjustResultCoordinates
import os
import datetime
import cv2
import time
import torch
from torchvision import datasets
from glob import glob
from skimage import io
import numpy as np
from PIL import Image

class ListDataset(torch.utils.data.Dataset):

    def __init__(self, path_list, canvas_size, mag_ratio):
        self.path_list = path_list
        self.list_length = len(path_list)
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.image_list, self.ratio_list, self.org_image_list = preprocess_img(self.path_list, self.canvas_size, self.mag_ratio)
    
    def __len__(self):
        return self.list_length

    def __getitem__(self, index):
        img = self.image_list[index]
        ratio = self.ratio_list[index]
        return img, ratio
    
    def get_org_img_list(self):
        return self.org_image_list

def post_process(y_list, ratio_list, text_threshold, link_threshold, low_text, poly_flag):
    result = []
    ratio_h = ratio_w = ratio_list[0]
    for y in y_list:
        polys_result = []
        score_text = y[:,:,0].cpu().data.numpy()
        score_link = y[:,:,1].cpu().data.numpy()
        boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly_flag)
        
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        for i, box in enumerate(polys):
            poly = np.array(box).astype(np.int32).reshape((-1))
            polys_result.append(poly)
        print('poly result ', len(polys_result))
        result.append(polys_result)
    print('post result ', len(result))
    return result
        

def batch_predict(detector, dataloaders, text_threshold, link_threshold, low_text, poly, device):
    overall_result = []

    with torch.no_grad():
        for image_tensors, ratio_tensors in dataloaders:
            
            x = image_tensors.to(device)
            
            begin = time.time()
            y, featrue = detector(x)
            end = time.time()
            print("detect time = ", end - begin)

            ratio = ratio_tensors.cpu().data.numpy()
            print('y ', y.size())
            print('ratio ', ratio_tensors)
            result = post_process(y, ratio, text_threshold, link_threshold, low_text, poly)
            begin_1 = time.time()
            overall_result = merge_list(overall_result, result)
            end_1 = time.time()
            print("append time = ", end_1 - begin_1)
    return overall_result

def merge_list(overall_result, result):
    for text_box in result:
        overall_result.append(text_box)
    return overall_result

def preprocess_img(path_list, canvas_size, mag_ratio):
    """
        preprocess image before detecting
        :param path_list: image file path list;
        :param canvas_size:
        :param mag_ratio:
        :return: img_list, ratio_list
    """
    img_list = []
    ratio_list = []
    org_image_list = []
    for img_path in path_list:
        img = io.imread(img_path)
        org_image_list.append(img)
        if img.shape[0] == 2: 
            img = img[0]
        if len(img.shape) == 2: 
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4:   
            img = img[:,:,:3]
        img = np.array(img)
        img_resized, target_ratio,_ = resize_aspect_ratio(img, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
        ratio = 1 / target_ratio
        ratio_list.append(ratio)
        x = normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        img_list.append(x)
    return img_list, ratio_list, org_image_list


def init_easyocr(model_path, context):
    file_name = model_path
    print(file_name)
    detector = get_detector(file_name, 'cuda') if context.lower() == 'cuda' else get_detector(file_name)
    print(detector)
    return detector

def load_data(path, b_size, workers):
    paths = glob(path)
    paths = sorted(paths)

    test_dataset = ListDataset(paths, 2560, 1.)

    org_image_list = test_dataset.get_org_img_list()

    pre_dataloaders = torch.utils.data.DataLoader(test_dataset, batch_size=b_size, shuffle=False, pin_memory = True, num_workers=workers)
    return pre_dataloaders, org_image_list