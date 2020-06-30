from daat import DAAT

from gluoncv import model_zoo, data
import mxnet as mx
import numpy as np

class gluoncv_annotator(DAAT):

    def model_setup(self):
        '''
        funtion to setup the model , the models classes and a color dict
        '''
    
        try:
            _ = mx.nd.array([1, 2, 3], ctx=mx.gpu(0))
            self.ctx = mx.gpu(0)
        except mx.MXNetError:
            self.ctx = mx.cpu()
        
#         optional
        self.threshold = 0.5
#         required
        self.model = model_zoo.get_model('faster_rcnn_fpn_syncbn_resnest101_coco',pretrained=True,ctx=self.ctx)
        self.model_classes = self.model.classes
        

    def predict(self,image): 
        '''
        function that takes an image runs predictions on the image the  returns a list of classes and bounding boxes

        args:
            image (numpy.ndarray) : image to run predictions on
        return:
            boxes (list) : list of bounding boxes in the form [[bb1_xmin,bb1_ymin,bb1_xmax,bb1_ymax],[bb2_xmin,bb2_ymin,bb2_xmax,bb2_ymax]] predicted from the image
            classes (list): list of classes predicted from the image
        '''
        x_rcnn, _ = data.transforms.presets.rcnn.transform_test(mx.nd.array(image))
        box_ids, scores, bboxes = self.model(x_rcnn.as_in_context(self.ctx))

        scores = scores.asnumpy().squeeze()
        class_ids = box_ids.asnumpy().squeeze()[np.where(scores>self.threshold)].astype(int)
        
        classes = np.array(fasterrcnn_coco_net.classes)[class_ids].tolist()
        bboxes = bboxes.asnumpy().squeeze()[np.where(scrs>0.5)].astype(int).tolist()

        return bboxes,classes
    
gluoncv_annotator().run()
