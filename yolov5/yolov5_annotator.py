from daat import DAAT


import torch
from torchvision import transforms as T
from PIL import Image

import numpy as np
from utils import google_utils
from utils.datasets import *
from utils.utils import *

class yolov5_annotator(DAAT):

    def model_setup(self):
        '''
        funtion to setup the model , the models classes and a color dict
        '''

        
        google_utils.attempt_download('yolov5x.pt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load('yolov5x.pt', map_location=self.device)['model'].float()
        self.model_classes = self.model.names 
      

    def predict(self,image): 
        '''
        function that takes an image runs predictions on the image the  returns a list of classes and bounding boxes

        args:
            image (numpy.ndarray) : image to run predictions on
        return:
            boxes (list) : list of bounding boxes in the form [[bb1_xmin,bb1_ymin,bb1_xmax,bb1_ymax],[bb2_xmin,bb2_ymin,bb2_xmax,bb2_ymax]] predicted from the image
            classes (list): list of classes predicted from the image
        '''

        img = letterbox(image,640)[0]
        img = Image.fromarray(img)
        transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
        img = transform(img) # Apply the transform to the image
        img = img.to(self.device)
        img = img.unsqueeze(0)

         
        pred = self.model(img)[0]
        pred = non_max_suppression(pred, 0.4, 0.5)
        
        bboxes = []
        classes = []
        for i, det in enumerate(pred):  # detections per image

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], np.array(image).shape).round()

                for *xyxy, conf, cls in det:
                    bboxes.append([int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])])
                    classes.append(self.model_classes[int(cls)])
                    

        return bboxes,classes
    
yolov5_annotator().run()
