from .utils import *
from .gui import *

from torchvision import models
from torchvision import transforms as T
from PIL import Image

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


class DAAT():
    def __init__(self):

        init_params = init_screen()
        self.image_dir = init_params['image_dir']
        self.image_extension = init_params['image_extension']
        self.edit_mode = init_params['edit_mode']
        self.do_augmentations = init_params['do_augmentations']
        self.convert_to_coco = init_params['convert_to_coco']
        self.test_train_split = init_params['test_train_split']
        self.test_size = init_params['test_size']
        self.ok = 0
        self.class_map = {}

        try:
            file = open(os.path.join(__location__, 'class_map'), 'rb')
            class_map = pickle.load(file)
            file.close()
        except FileNotFoundError:
            class_map = {}
        #         setup model
        self.model_classes = []  # this needs to be set in model setup
        self.color_dict = {}  # dictionary mapping classes to colors
        self.model_setup()

        if not class_map:
            if not self.class_map:
                self.class_map = {(i + 48): self.model_classes[i] for i in range(10)}
        elif class_map:
            self.class_map = class_map

        #         run checks to see if annotations have been done to enable the tool to continue from where annotations were left off
        #         creates directories
        #         also allows re-labeling if opened in edit mode
        self.run_checks()

        #         annotation specific variables
        self.select = False
        self.temp = list()
        self.state = list()

    def run_checks(self, ):

        non_native_classes = set(self.class_map.values()).difference(set(self.model_classes))
        if non_native_classes:
            self.model_classes += list(non_native_classes)
            self.color_dict = dict(zip(self.model_classes, get_distinct_colors(len(self.model_classes))))
        else:
            self.color_dict = dict(zip(self.model_classes, get_distinct_colors(len(self.model_classes))))

        self.annotations_dir = f'{self.image_dir}/annotations'
        check_dir(self.annotations_dir)

        self.skipped_dir = f'{self.image_dir}/skipped'
        check_dir(self.skipped_dir)

        image_paths = glob(f'{self.image_dir}/*{self.image_extension}')

        annotations_paths = glob(f'{self.annotations_dir}/*.xml')
        skipped_paths = glob(f'{self.skipped_dir}/*.txt')

        print(len(image_paths), ' images in directory')
        print(len(annotations_paths), ' annotated in directory')
        print(len(skipped_paths), ' skipped in directory')

        get_name = lambda x: x.split('.')[0]
        get_filename = lambda x: os.path.split(x)[1]

        give_extention = lambda x: f'{x}{self.image_extension}'
        give_path = lambda x: os.path.join(self.image_dir, x)

        if self.edit_mode:
            image_names = map(get_filename, annotations_paths + skipped_paths)
            images_to_annotate = map(get_name, image_names)
            images_to_annotate = map(give_path, images_to_annotate)
            self.images_to_annotate = list(map(give_extention, images_to_annotate))
            print(len(self.images_to_annotate), ' images to edit')
        else:
            annotated_image_names = map(get_filename, annotations_paths + skipped_paths)
            all_images_names = map(get_filename, image_paths)
            images_to_annotate = set(map(get_name, all_images_names)).difference(
                set(map(get_name, annotated_image_names)))
            images_to_annotate = map(give_path, images_to_annotate)
            self.images_to_annotate = list(map(give_extention, images_to_annotate))
            print(len(self.images_to_annotate), ' images to annotate')

        if len(self.images_to_annotate) > 0:
            self.ok = 1
        else:
            sg.popup('No images to annotate detected, please check image directory or image extension')

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

        self.color_dict = dict(zip(self.model_classes, get_distinct_colors(len(self.model_classes))))

    def predict(self, image):
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
        transform = T.Compose([T.ToTensor()])  # Defing PyTorch Transform
        img = transform(image)  # Apply the transform to the image
        pred = model([img])  # Pass the image to the model
        pred_class = [self.model_classes[i] for i in list(pred[0]['labels'].numpy())]  # Get the Prediction Score
        pred_boxes = pred[0]['boxes'].detach().numpy().astype(int).tolist()  # Bounding boxes
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][
            -1]  # Get list of index with score greater than threshold.
        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]

        return pred_boxes, pred_class

    def post_routines(self, ):
        file = open(f'{self.image_dir}/class_info.txt', 'w')
        class_report = get_class_report(self.annotations_dir)
        file.writelines(class_report)
        file.close()

        classes = list(map(lambda x: " ".join(x.strip().split()[:-1]), class_report))

        if self.convert_to_coco:
            voc2coco(self.annotations_dir, classes, self.do_augmentations, self.test_train_split, self.test_size)

        elif not self.convert_to_coco and self.do_augmentations:

            classes_dict = {v: k for k, v in enumerate(classes)}
            xml_files = glob(f'{self.annotations_dir}/*.xml')

            if self.test_train_split:
                train_xml_files, test_xml_files = get_test_train_split(xml_files, classes, classes_dict, self.test_size)
                _ = get_augmentations(xml_files, classes_dict, self.annotations_dir)
            #                             TODO
            else:
                _ = get_augmentations(xml_files, classes_dict, self.annotations_dir)

    def annotate(self, image_path):
        '''
        1. loads image
        2. checks if xml exists , indication of annotated
        3. loads annotations otherwise predict boxes
        4. allow annotations

        '''

        image = cv2.imread(image_path)

        image_dir, filename = os.path.split(image_path)
        check_xml = f'{self.image_dir}/annotations/{filename.split(".")[0]}.xml'

        if os.path.isfile(check_xml):
            classes, boxes, _ = read_content(check_xml)
            classes = np.array(classes)
            boxes = np.array(boxes)

        else:
            orig_boxes, orig_classes = self.predict(image)
            boxes, classes = orig_boxes, orig_classes

        labels, counts = np.unique(classes, return_counts=True)
        detections = map(lambda x: f'{x[0]}:{x[1]}', zip(labels, counts))
        detections_string = f'detected:{list(detections)}'

        show_preds_toggle = False
        instance_buffer = 'car'
        cycle_counter = -1

        annotations = list()
        annotated_classes = list()

        self.zoom_box = np.zeros((300, 200, 3), np.uint8)
        zoom_box_radius = 150

        def annotate_object_callback(event, x, y, flags, params):

            self.zoom_box = get_zoom_box(image, x, y, zoom_box_radius)

            if event == cv2.EVENT_LBUTTONDOWN and not self.select:

                np.copyto(buffer_image, self.state)
                self.state = buffer_image.copy()
                self.temp.append((x, y))
                self.select = True

            elif event == cv2.EVENT_LBUTTONUP and self.select:

                if self.temp[0] != (x, y):
                    annotations.append([*self.temp[0], x, y])
                    annotated_classes.append(instance_buffer)
                    cv2.rectangle(buffer_image, self.temp[0], (x, y),
                                  tuple(self.color_dict.get(instance_buffer, [255, 255, 255])), 1)

                self.temp = []
                self.select = False
                self.state = buffer_image.copy()

            elif self.select:

                if self.temp[0] != (x, y):
                    np.copyto(buffer_image, self.state)
                    cv2.rectangle(buffer_image, self.temp[0], (x, y),
                                  tuple(self.color_dict.get(instance_buffer, [255, 255, 255])), 1)
            else:

                np.copyto(buffer_image, self.state)
                cv2.line(buffer_image, (0, y), (buffer_image.shape[1], y), (255, 255, 255), 1)  # verticle line
                cv2.line(buffer_image, (x, 0), (x, buffer_image.shape[0]), (255, 255, 255), 1)  # horizontal line

        buffer_image = image.copy()
        self.state = buffer_image.copy()  # need this to draw cross hairs without persisting

        window_name = "Assisted Annotation Tool"
        info_window = 'info'

        screen_w, screen_h = get_screen_size()

        cv2.namedWindow(window_name)
        cv2.namedWindow(info_window)

        cv2.moveWindow(window_name, 2 * screen_w // 3, 0)
        cv2.moveWindow(info_window, 2 * screen_w // 3, buffer_image.shape[0] + 50)

        cv2.setMouseCallback(window_name, annotate_object_callback)

        while 1:

            cv2.imshow(window_name, buffer_image)
            cv2.imshow(info_window, draw_info(self.class_map, self.zoom_box, instance_buffer, detections_string))
            #             cv2.imshow('zoombox',self.zoom_box)
            key = cv2.waitKey(1)
            if key == 13:

                print("DONE")
                final_annotations = boxes.tolist() + annotations
                final_classes = classes.tolist() + annotated_classes
                cv2.destroyAllWindows()
                return final_annotations, final_classes

            elif key == ord('r'):

                print("RESET")
                buffer_image = image.copy()
                orig_boxes, orig_classes = self.predict(image)
                boxes, classes = orig_boxes, orig_classes
                annotations = list()
                annotated_classes = list()

            elif key == ord('u'):

                if annotations:
                    annotations.pop()
                    annotated_classes.pop()
                    buffer_image = image.copy()
                    for i, box in enumerate(annotations):
                        cv2.rectangle(buffer_image, tuple(box[:2]), tuple(box[2:]),
                                      tuple(self.color_dict.get(annotated_classes[i], [255, 255, 255])))

                    if show_preds_toggle:
                        for i, box in enumerate(boxes):
                            cv2.rectangle(buffer_image, tuple(box[:2]), tuple(box[2:]),
                                          tuple(self.color_dict.get(classes[i], [255, 255, 255])))

                    self.state = buffer_image.copy()

            elif key > 47 and key < 58:
                #             change class
                try:
                    instance_buffer = self.class_map[key]
                except KeyError:
                    sg.popup('key not assigned')

            elif key == 61 or key == 41:

                diff_class = select_classes_screen(self.model_classes)
                instance_buffer = diff_class if diff_class else instance_buffer

            elif key == 81 or key == 83:

                if key == 81:
                    return -3
                elif key == 83:
                    return -4

            elif key == ord('n'):

                new_class = add_new_class_screen()
                if new_class not in ('', None):
                    instance_buffer = new_class
                    self.model_classes.append(new_class)
                    self.color_dict = dict(zip(self.model_classes, get_distinct_colors(len(self.model_classes))))

            elif key == ord('p'):
                #             toggle predictions all predictions
                buffer_image = image.copy()
                show_preds_toggle = not show_preds_toggle
                if show_preds_toggle:
                    for i, box in enumerate(boxes):
                        cv2.rectangle(buffer_image, tuple(box[:2]), tuple(box[2:]),
                                      tuple(self.color_dict.get(classes[i], [255, 255, 255])))

                for i, box in enumerate(annotations):
                    cv2.rectangle(buffer_image, tuple(box[:2]), tuple(box[2:]),
                                  tuple(self.color_dict.get(annotated_classes[i], [255, 255, 255])))

                self.state = buffer_image.copy()
                cycle_counter = -1

            elif key == ord('c'):

                buffer_image = image.copy()

                cycle_boxes = boxes.astype(int).tolist() + annotations
                cycle_classes = classes.tolist() + annotated_classes

                if not len(cycle_boxes):
                    continue

                #                 for i,box in enumerate(annotations):
                #                     cv2.rectangle(buffer_image,tuple(box[:2]),tuple(box[2:]), tuple( self.color_dict.get(annotated_classes[i],[255,255,255])) )

                if cycle_counter == (len(cycle_boxes) - 1):
                    cycle_counter = -1

                cycle_counter += 1
                cv2.rectangle(buffer_image, tuple(cycle_boxes[cycle_counter][:2]),
                              tuple(cycle_boxes[cycle_counter][2:]),
                              tuple(self.color_dict.get(cycle_classes[cycle_counter], [255, 255, 255])), )

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_color = (255, 255, 255)
                thickness = 2

                text_width, text_height = \
                cv2.getTextSize(cycle_classes[cycle_counter], font, fontScale=font_scale, thickness=thickness)[0]
                bb_x, bb_y = tuple(cycle_boxes[cycle_counter][:2])
                box_coords = ((int(bb_x - 2), int(bb_y - text_height - 2)), (int(bb_x + text_width + 2), int(bb_y + 2)))

                cv2.rectangle(buffer_image, box_coords[0], box_coords[1],
                              tuple(self.color_dict.get(cycle_classes[cycle_counter], [255, 255, 255])), cv2.FILLED)

                if font_color == tuple(self.color_dict.get(cycle_classes[cycle_counter], [255, 255, 255])):
                    #                     ensure font color and box color is never the same
                    import random
                    font_color = tuple(
                        random.shuffle(self.color_dict.get(cycle_classes[cycle_counter], [255, 255, 255])))

                cv2.putText(buffer_image, cycle_classes[cycle_counter], tuple(cycle_boxes[cycle_counter][:2]), font,
                            font_scale, font_color, thickness)

            elif key == ord('a'):

                class_map = assign_hotkey_screen(self.model_classes)
                self.class_map = class_map if class_map else self.class_map


            elif key == ord('d'):

                if cycle_counter != -1:

                    if cycle_counter <= (len(boxes) - 1):
                        boxes = np.delete(boxes, cycle_counter, axis=0)
                        classes = np.delete(classes, cycle_counter)
                    elif annotations and cycle_counter > (len(boxes) - 1):
                        annotations = np.delete(annotations, cycle_counter - len(boxes), axis=0).tolist()
                        annotated_classes = np.delete(annotated_classes, cycle_counter - len(boxes)).tolist()

                    buffer_image = image.copy()
                    for i, box in enumerate(annotations):
                        cv2.rectangle(buffer_image, tuple(box[:2]), tuple(box[2:]),
                                      tuple(self.color_dict.get(annotated_classes[i], [255, 255, 255])))
                    cycle_counter = -1
                    print('PREDICTION DELETED')
                    self.state = buffer_image.copy()

            elif key == ord('s'):
                return -2

            elif key == 27:
                #             exit
                file = open(os.path.join(__location__, 'class_map'), 'wb')
                pickle.dump(self.class_map, file)
                file.close()
                cv2.destroyAllWindows()
                return -1

    def run(self, ):

        image_index = 0

        while self.ok:

            image_path = self.images_to_annotate[image_index]
            result = self.annotate(image_path)
            if isinstance(result, int):

                if result == -1:
                    self.post_routines()
                    return -1

                elif result == -2:
                    filename = os.path.basename(image_path).split('.')[0]
                    open(f'{self.skipped_dir}/{filename}.txt', 'w').close()
                    image_index += 1

                elif result == -3:
                    image_index -= 1

                elif result == -4:
                    image_index += 1

            elif isinstance(result, tuple):
                final_annotations, final_classes = result
                write2voc(image_path, final_annotations, final_classes, self.annotations_dir)
                image_index += 1

            if (image_index == len(self.images_to_annotate)) or (image_index == -len(self.images_to_annotate)):
                if self.edit_mode:
                    image_index = 0
                else:
                    sg.popup('All images Annotates , well done !!!')
                    self.post_routines()
                    return -1
