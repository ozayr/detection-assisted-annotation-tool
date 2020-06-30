from daat import DAAT

from torchvision import models
from torchvision import transforms as T
from PIL import Image

class pytorch_annotator(DAAT):

    def model_setup(self):
        '''
        funtion to setup the model , the models classes and a color dict
        '''
    #         default setting using opencv object detection tutorial which uses pytorch
      
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        
        self.model_classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        




    def predict(self,image): 
        '''
        function that takes an image runs predictions on the image the  returns a list of classes and bounding boxes

        args:
            image (numpy.ndarray) : image to run predictions on
        return:
            boxes (list) : list of bounding boxes in the form [xmin,ymin,xmax,ymax] predicted from the image
            classes (list): list of classes predicted from the image
        '''

        image = Image.fromarray(image)
        threshold = 0.5
        transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
        img = transform(image) # Apply the transform to the image
        pred = self.model([img]) # Pass the image to the model
        pred_class = [self.model_classes[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
        pred_boxes = pred[0]['boxes'].detach().numpy().astype(int).tolist() # Bounding boxes
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]

        return pred_boxes,pred_class


pytorch_annotator().run()
