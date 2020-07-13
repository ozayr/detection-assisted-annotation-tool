# Detection Assisted Annotation Tool

Labeling tool used to label images for object detection, with assistance from an object detector of your choice by overriding the model_setup and predict function (see example , yolo,detectron,gluon-cv).

Saves images by default in pascalVOC format in an annotations folder created by the tool in the images directory. Allows annotations to be saved in COCO format by selecting the appropriate option from the startup menu. if test_train_split enabled, the tool can split the annotations based on a stratified split of classes or a sequential split provided the images have some sort of sequential numbering eg 1,2,3,4/01,02,03/1002,1003,1004...

TODO:// Demo images 


 


 
