from daat import DAAT
import os
import numpy as np

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

class detectron_annotator(DAAT):
    def model_setup(self):
        '''
        funtion to setup the model , the models classes and a color dict
        '''

        config_file_path =  'detectron2/configs/FCRCNN_ResNeSt200_config.yaml'
        model_weights_path = 'detectron2/models/FCRCNN_ResNeSt200_FPN.pth'

        print('using Config file:',os.path.basename(config_file_path))
        print('using Model Weights file:',os.path.basename(model_weights_path))

        cfg = get_cfg()
        cfg.merge_from_file(config_file_path)
        cfg.MODEL.WEIGHTS = model_weights_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        self.predictor = DefaultPredictor(cfg)

        self.model_classes  = self.predictor.metadata.thing_classes
       
        self.class_map = {48: 'person',49: 'car',50: 'motorcycle',51: 'airplane',52: 'bus',53: 'truck', 54: 'traffic light',55: 'bill board',56: 'sign',57:'light pole'}


    def predict(self,image): 
        '''
        function that takes an image runs predictions on the image the  returns a list of classes and bounding boxes

        args:
            image (numpy.ndarray) : image to run predictions on
        return:
            boxes (list) : list of bounding boxes in the form [xmin,ymin,xmax,ymax] predicted from the image
            classes (list): list of classes predicted from the image
        '''

        predictions = self.predictor(image)
        classes = np.array(self.predictor.metadata.thing_classes)[predictions['instances'].get_fields()['pred_classes'].cpu().numpy()]
        boxes = np.round(predictions['instances'].get_fields()['pred_boxes'].tensor.cpu().numpy())


        return boxes,classes
    
detectron_annotator().run()
