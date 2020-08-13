from detection import init_easyocr, load_data, batch_predict
from cnocr import CnOcr
from cnocr.detect_result import *
import time
from glob import glob

detector = init_easyocr(model_path='./model/craft_mlt_25k.pth', context='cuda')
recognizer = CnOcr(model_name='densenet-lite-gru', context='gpu')

pre_dataloaders, org_image_list = load_data('./data/46_5167189/*', 8, 20)

begin = time.time()

text_boxs = batch_predict(detector, pre_dataloaders, 0.7, 0.1, 0.4, False, 'cuda')

result = []
for img, text_box in zip(org_image_list, text_boxs):
    for rect in text_box:
        if rect[1] >= rect[5] or rect[0] >= rect[4]:
            continue
        part_img = img[rect[1]:rect[5], rect[0]:rect[4],:]
        
        res = recognizer.ocr_for_single_line(part_img)
        
        if res == None:
            continue
        result.append(res.result)
    # print(result)
end = time.time()
print("gpu times = ", end - begin)
print(result)