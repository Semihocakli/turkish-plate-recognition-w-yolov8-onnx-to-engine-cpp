import cv2
import os
import torch
import numpy as np

from ultralytics.utils import ops
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results

def get_text_color(box_color):
    text_color = (255,255,255)

    brightness = box_color[2]*0.299 + box_color[1]*0.587 + box_color[0]*0.114

    if(brightness > 180):
        text_color = (0, 0, 0)

    return text_color

def draw_box(img, detection_output, class_list, colors) :    
    # Copy image, in case that we need original image for something
    out_image = img 

    for run_output in detection_output :
        # Unpack
        label, con, box = run_output        

        # Choose color
        box_color = colors[int(label.item())]
        # text_color = (255,255,255)
        text_color = get_text_color(box_color)
        # Get Class Name
        label = class_list[int(label.item())]
        # Draw object box
        first_half_box = (int(box[0].item()),int(box[1].item()))
        second_half_box = (int(box[2].item()),int(box[3].item()))
        cv2.rectangle(out_image, first_half_box, second_half_box, box_color, 2)
        # Create text
        text_print = '{label} {con:.2f}'.format(label = label, con = con.item())
        # Locate text position
        text_location = (int(box[0]), int(box[1] - 10 ))
        # Get size and baseline
        labelSize, baseLine = cv2.getTextSize(text_print, cv2.FONT_HERSHEY_SIMPLEX, 1, 1) 
        
        # Draw text's background
        cv2.rectangle(out_image 
                        , (int(box[0]), int(box[1] - labelSize[1] - 10 ))
                        , (int(box[0])+labelSize[0], int(box[1] + baseLine-10))
                        , box_color , cv2.FILLED)        
        # Put text
        cv2.putText(out_image, text_print ,text_location
                    , cv2.FONT_HERSHEY_SIMPLEX , 1
                    , text_color, 2, cv2.LINE_AA)

    return out_image

def draw_fps(avg_fps, combined_img):        
    avg_fps_str = float("{:.2f}".format(avg_fps))
    
    cv2.rectangle(combined_img, (10,2), (660,110), (255,255,255), -1)
    cv2.putText(combined_img, "FPS: "+str(avg_fps_str), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0,255,0), thickness=6)

    return combined_img

def get_name(source_path):    
    name_idx = 0
    file_pos = (source_path).rfind('\\')

    if(file_pos == -1):
        file_pos = (source_path).rfind('/')

        if(file_pos == -1):
            file_pos = 0
    
    name_idx = file_pos + 1

    name = source_path[name_idx:]

    return name

def get_save_path(file_name, folder_name):
    path = "result"
    save_path = os.path.join(path, folder_name)

    exists = os.path.exists(save_path) 

    if(not exists):
        os.makedirs(save_path)

    save_path = os.path.join(save_path, file_name)    

    return save_path

def preprocess(img):   
    # LetterBox
    im = LetterBox((640, 640), False)(image=img)
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    img = torch.from_numpy(im).to(torch.device('cuda:0'))
    img = img.half()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0

    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    return img

def postprocess(preds, img, orig_img, names, source_path):
    preds = ops.non_max_suppression(preds,
                                    0.5,
                                    0.7,
                                    agnostic=False,
                                    max_det=300,
                                    classes=None)

    results = []
    for i, pred in enumerate(preds):
        orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
        shape = orig_img.shape
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
        path = source_path
        img_path = path[i] if isinstance(path, list) else path
        results.append(Results(orig_img=orig_img, path=img_path, names=names, boxes=pred))

    return results

# def letterbox(img):
    

#     return im

