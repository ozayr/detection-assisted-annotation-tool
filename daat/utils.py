# UTILITTIES
import random
import tkinter as tk
from pascal_voc_writer import Writer

import numpy as np
import cv2
import os
from glob import glob
import json

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
    
from skmultilearn.model_selection import IterativeStratification

from bbaug.policies import policies
import imgaug.augmenters as iaa


def get_screen_size():
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    return screen_width,screen_height

def get_distinct_colors(n):
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r,g,b)) 
    random.shuffle(ret)
    return ret


def draw_info(class_map,zoombox,instance_buffer,detections):
    
    info_image = np.zeros((300,900,3),np.uint8)
    
    class_map_dict = {i:item  for i,item in enumerate(class_map.values())}
    static_string = f'hot key assigned classes:\n{class_map_dict}\nensure instance selected before drawing box\n\n{"done":25}: Enter\n{"reset":25}: r\n{"cycle predictions":25}: c\n{"delete cycled prediction":25}: d\n{"show/hide predictions":25}: p\n{"undo boxes drawn":25}: u\n{"skip image":25}: s\n{"add class":25}: n\n{"select from class list":25}: +\n{"reassign hot keys":25}: a\n\nTIP: show predictions , cycle , delete if need , then manually annotate by dragging box over ROI\nEXIT: Esc'
    label_status = f'now labeling: {instance_buffer}'
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    font_scale              = 0.5
    font_color              = (255,255,255)
    line_type               = 1
    _, text_height = cv2.getTextSize('test_text', font, font_scale, line_type)[0]
    
    line_counter = 0
    for line in static_string.split('\n'):
        
        w, _ = cv2.getTextSize(line, font, font_scale, line_type)[0]
        if w > info_image.shape[1]:
            
            cv2.putText(info_image , line[:len(line)//2] ,(10,(text_height+1)*line_counter + 10 ) , font , font_scale , font_color , line_type)
            line_counter += 1
            line = line[len(line)//2:]
           
        
        cv2.putText(info_image , line ,(10,(text_height+1)*line_counter + 10 ) , font , font_scale , font_color , line_type)
        line_counter += 1
    
    cv2.putText(info_image,'' ,(10,(text_height+1)*line_counter + 10 ) , font , font_scale , font_color , line_type)
    line_counter += 1
    
    cv2.putText(info_image , detections ,(10,(text_height+1)*line_counter + 10 ) , font , font_scale , font_color , line_type)
    line_counter += 1
    cv2.putText(info_image , label_status ,(10,(text_height+1)*line_counter + 10 ) , font , font_scale , font_color , line_type)
   
    zoombox = cv2.copyMakeBorder(zoombox,0,300-zoombox.shape[0],0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
    info_image = cv2.hconcat([info_image,zoombox])
    
    return info_image

def pad_image(image,x,y,radius):
    h,w,_ = image.shape
    if (y-radius < 0):
        image = cv2.copyMakeBorder( image, abs(y-radius), 0, 0, 0, cv2.BORDER_CONSTANT,value = [0,0,0])
        y += abs(y-radius)
    if (x-radius < 0 ):
        image = cv2.copyMakeBorder( image, 0, 0, abs(x-radius), 0, cv2.BORDER_CONSTANT,value = [0,0,0])
        x += abs(x-radius)
    if (x+radius>w):
        image = cv2.copyMakeBorder( image, 0, 0, 0, abs(w-(x+radius)), cv2.BORDER_CONSTANT,value = [0,0,0])
    if (y+radius>h):
        image = cv2.copyMakeBorder( image, 0,  abs(h-(x+radius)), 0,0, cv2.BORDER_CONSTANT,value = [0,0,0])
    return image,x,y

def get_zoom_box(image,x,y,radius = 100):
    
    h,w,_ = image.shape
    if (y-radius < 0 ) or (x-radius < 0) or (y+radius > h) or (x+radius > w):
        image,x,y = pad_image(image,x,y,radius)
    
    box = image[y-radius:y+radius,x-radius:x+radius]
    zoombox = cv2.resize(box, None, fx= 1.8, fy= 1.8, interpolation= cv2.INTER_LINEAR)
    zoombox = zoombox[zoombox.shape[0]//2-radius:zoombox.shape[0]//2+radius,zoombox.shape[1]//2-radius:zoombox.shape[1]//2+radius]
    cv2.circle(zoombox,(zoombox.shape[0]//2,zoombox.shape[1]//2),2,[255,255,0],-1)
    
    return zoombox




def get_aug_policy():
    
    def flip(magnitude: int) -> iaa.Fliplr:
        return iaa.Fliplr()

    def translate(magnitude: int) -> iaa.Affine:
        return iaa.Affine(translate_percent={'x': (-magnitude,magnitude) , 'y' :(-magnitude,magnitude)})

    def scale(magnitude: float) -> iaa.Affine:
        return iaa.Affine(scale=(1-magnitude, 1+magnitude))

    def rotate(magnitude: int) -> iaa.Affine:
        return iaa.Affine(rotate=(-magnitude, magnitude))

    def motion(magnitude: int) -> iaa.blur:
        return iaa.MotionBlur(k=magnitude, angle=90)


    aug_policy_container = policies.PolicyContainer(
        [
            [policies.POLICY_TUPLE(name='flip', probability=1.0, magnitude=1)],
            [policies.POLICY_TUPLE(name='translate', probability=1.0, magnitude=20)],
            [policies.POLICY_TUPLE(name='scale', probability=1.0, magnitude=0.4)],
#             [policies.POLICY_TUPLE(name='rotate', probability=1.0, magnitude=30)],
#             [policies.POLICY_TUPLE(name='motion', probability=1.0, magnitude=7)]

        ],
        name_to_augmentation={
            'flip': flip,
            'translate': translate,
            'scale':scale,
#             'rotate':rotate,
#             'motion':motion
        }
    )
    return aug_policy_container


def get_augmentations(xml_files,classes_dict,annotations_dir):
    
    aug_image_dir =  os.path.join('/'.join(annotations_dir.split('/')[:-1]),'augmented_images')
    aug_xml_dir =  os.path.join('/'.join(annotations_dir.split('/')[:-1]),'augmented_annotations')
    
    check_dir(aug_image_dir)
    check_dir(aug_xml_dir)
    
    for f,x in zip(glob(f'{aug_image_dir}/*.jpg'),glob(f'{aug_xml_dir}/*.xml')):
        os.remove(f)
        os.remove(x)
    
    
    policy_container = get_aug_policy()
    aug_xml_files = []
                   
    for xml_file in xml_files:
        categories,boxes,image_path = read_content(xml_file)
        labels = [classes_dict[label] for label in categories]
        img = cv2.imread(image_path)
        
        # Select a random sub-policy from the policy list
        random_policy = policy_container.select_random_policy()
        # Apply this augmentation to the image, returns the augmented image and bounding boxes
        # The boxes must be at a pixel level. e.g. x_min, y_min, x_max, y_max with pixel values
        img_aug, bbs_aug = policy_container.apply_augmentation(random_policy,img,boxes,labels)
        # Only return the augmented image and bounded boxes if there are
        # boxes present after the image augmentation
        if bbs_aug.size > 0:
            image_name = f'{aug_image_dir}/aug_{os.path.basename(image_path)}'
            cv2.imwrite(image_name,img_aug)      
            xml_file_name = write2voc(image_path,bbs_aug,categories,aug_xml_dir,pre_fix='aug_')
            aug_xml_files.append(xml_file_name)
        
    return aug_xml_files


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


def get_categories(xml_files):
    """Generate category name to id mapping from a list of xml files.
    
    Arguments:
        xml_files {list} -- A list of xml file paths.
    
    Returns:
        dict -- category name to id mapping.
    """
    classes_names = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            classes_names.append(member[0].text)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return {name: i for i, name in enumerate(classes_names)}


def write_to_coco(xml_files, json_file,categories):
    
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    bnd_id = 1
    
    for image_id,xml_file in enumerate(xml_files):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        path = get(root, "path")
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
            coco_url = path[0].text
        elif len(path) == 0:
            filename = get_and_check(root, "filename", 1).text
            coco_url = path[0].text
        else:
            raise ValueError("%d paths found in %s" % (len(path), xml_file))

        image_id = image_id
        size = get_and_check(root, "size", 1)
        width = int(get_and_check(size, "width", 1).text)
        height = int(get_and_check(size, "height", 1).text)
        image = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id,
            "coco_url":path[0].text
        }
        json_dict["images"].append(image)
        ## Currently we do not support segmentation.
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, "object"):
            category = get_and_check(obj, "name", 1).text
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, "bndbox", 1)
            xmin = int(get_and_check(bndbox, "xmin", 1).text) - 1
            ymin = int(get_and_check(bndbox, "ymin", 1).text) - 1
            xmax = int(get_and_check(bndbox, "xmax", 1).text)
            ymax = int(get_and_check(bndbox, "ymax", 1).text)
            #assert xmax > xmin
            if xmin > xmax:
               xmin,xmax = xmax,xmin
            if ymin > ymax:
               ymin,ymax = ymax,ymin
            #assert ymax > ymin
            
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {
                "area": o_width * o_height,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [xmin, ymin, o_width, o_height],
                "category_id": category_id,
                "id": bnd_id,
                "ignore": 0,
                "segmentation": [],
            }
            json_dict["annotations"].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


    
def get_test_train_split(xml_files,classes,classes_dict,test_size):
    
    label_array = np.zeros((len(xml_files),len(classes)))
    
    for i,file in enumerate(xml_files):
       labels,_,_ = read_content(file)
       sparse_labels = list(map(lambda x : classes_dict[x] ,labels))
       label_array[i,sparse_labels] = 1

    kf = IterativeStratification(n_splits = int(1/test_size) )
    train,test = next(kf.split(xml_files,label_array))
    
    train_xml_files = np.array(xml_files)[train].tolist()
    test_xml_files = np.array(xml_files)[test].tolist()
    
    return train_xml_files,test_xml_files
    

def voc2coco(annotations_dir,classes,do_augmentations,test_train_split=False,test_size = 0.25):
    '''
    function does a stratified split between all classes into test and train sets and saves as coco .json format
    args:
        annotations_dir: directory to xml annotations
        classes: all classes present in the annotated images
        do_augmentations: flag that enables augmentations to only the train split
                        
        augmentations are done the saved in a seperate folder in the image directory.
        augmentations are done using imgaug and bbaug.
        a random augmentation is done for all annotated images in the train set 
                        
        
    '''
    
    classes_dict = {v:k for k,v in enumerate(classes)}
    #PRE_DEFINE_CATEGORIES = {k:v for k,v in enumerate(classes)}

    xml_files = glob(f'{annotations_dir}/*.xml')
    
    if (len(xml_files) >= int(1/test_size)) and test_train_split:
        
        
        train_xml_files,test_xml_files = get_test_train_split(xml_files,classes,classes_dict,test_size)
        # take train xml files add augmentations get aug xml files add to train xml files
        if do_augmentations:
            train_xml_files += get_augmentations(train_xml_files,classes_dict,annotations_dir)

        
        print(f"Number of xml files: {len(xml_files)}")

        print(f"test split xml files: {len(test_xml_files)}")
        test_class_report = get_class_report(xml_files = test_xml_files)
        print(list(map(lambda x:  " ".join(x.strip().split()[:-1]) ,test_class_report)))
        
        print(f"train split xml files: {len(train_xml_files)}")
        train_class_report = get_class_report(xml_files = train_xml_files) 
        print(list(map(lambda x:  " ".join(x.strip().split()[:-1]) ,train_class_report)))

        write_to_coco(test_xml_files, os.path.join(annotations_dir,'coco_test.json'),classes_dict)
        write_to_coco(train_xml_files, os.path.join(annotations_dir,'coco_train.json'),classes_dict)
        print('Conversion to coco done')
    elif not (len(xml_files) >= int(1/test_size)) and test_train_split:
        print(f'please annotate more than {int(1/test_size)} images for the specified test_size of {test_size},coco conversion not done')
    
    else:
         write_to_coco(xml_files, os.path.join(annotations_dir,'coco.json'),classes_dict)
        
        
def get_class_report(annotations_dir=None,xml_files = None):
    '''
    generate a report of the classes and their counts
    
    args:
        annotations_dir : path to folder that contains the xml files
    '''
    
    classes = []
    if xml_files:
        annotations_paths = xml_files
    else:
        annotations_paths = glob(f'{annotations_dir}/*.xml')
    
    for each_xml_file in annotations_paths:
        tree = ET.parse(each_xml_file)
        root = tree.getroot()
        for child in root:
            if child.tag == 'object':
                for item in child:
                    if item.tag == 'name':
                        classes.append(item.text)
                        
    cls,counts = np.unique(classes,return_counts=True)
    classes = []
    
    for cls,count in zip(cls,counts):
        classes.append(f'{cls} {count}\n')
        
    return classes

def read_content(xml_file):
    '''
    get class names and bounding boxes 

    args:
    xml_file: path to xml file
    '''
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    class_names = []
    image_path = root.find('path').text

    for boxes in root.iter('object'):

        classes = boxes.find('name').text

        ymin, xmin, ymax, xmax = None, None, None, None

        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)
        class_names.append(classes)

    return class_names,list_with_all_boxes,image_path

def display_all_classes(classes_list):
    '''
    display all classes formatted
    '''
    for i,cls in enumerate(classes_list):
        if i%2==0:
            print()
        print(f'{str(cls):25}',end='')

def check_dir(directory):
    '''
    check of dictionary exists, if not create it
    '''
    if not os.path.exists(directory):
        os.mkdir(directory)
        print("Directory " , directory ,  " Created ")
    else:    
        print("Directory " , directory ,  " already exists")

def write2voc(image_path,final_annotations,final_classes,xml_dir,pre_fix=''):
    '''
    create voc formatted xml file from image objects and bounding boxes
    '''
    image_shape = cv2.imread(image_path).shape
    writer = Writer(image_path, image_shape[1], image_shape[0])
    
    for annotation,label in zip(final_annotations,final_classes):
        annotation = list(map(int,annotation))
        writer.addObject(label, *annotation)
        
    filename = os.path.basename(image_path).split('.')[0]
    xml_file_name = f'{xml_dir}/{pre_fix}{filename}.xml'
    writer.save(xml_file_name)
    
    return xml_file_name




