import os
import cv2
import random
from matplotlib import image, transforms
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.cfg_config import cfg

class Dataset(object):
    def __init__(self,is_training:bool,dataset_type:str="converted_coco"):
        self.strides,self.anchors,NUM_CLASS,XY_SCALE = utils.load_config(tiny=True)
        self.dataset_type= dataset_type
        self.annot_path = (
            cfg.TRAIN.ANNOT_PATH if is_training else cfg.TEST.ANNOT_PATH
        )
        self.input_sizes = (
            cfg.TRAIN.INPUT_SIZE if is_training else cfg.TEST.INPUT_SIZE
        )
        self.batch_size = (
            cfg.TRAIN.BATCH_SIZE if is_training else cfg.TEST.BATCH_SIZE
        )
        self.data_aug = cfg.TRAIN.DATA_AUG if is_training else cfg.TEST.DATA_AUG
        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.classes = utils.read_class_name(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations()
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0
    def __len__(self):
        return self.num_batchs
    def load_annotations(self):
        with open(self.annot_path,"r") as f:
            txt = f.readlines()
            if self.dataset_type == "converted_coco":
                annotations = [
                    line.strip() for line in txt if len(line.strip().split()[1:]) != 0
                ]
        np.random.shuffle(annotations)
        return annotations
    def parse_annotation(self,annotation):
        classes=utils.read_class_name(cfg.YOLO.CLASSES)
        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError(f"{image_path} does not exist")
        image = cv2.imread(image_path)
        bboxes = np.array(
            [list(map(int,box.split(","))) for box in line[1:]]
        )
        if self.data_aug:
            image , bboxes = self.random_horizontal_flip(np.copy(image),np.copy(bboxes))
            image , bboxes = self.random_crop(np.copy(image),np.copy(bboxes))
            image , bboxes = self.random_translate(np.copy(image),np.copy(bboxes))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        image, bboxes = utils.image_preprocess(
            np.copy(image),
            [self.train_input_size, self.train_input_size],
            np.copy(bboxes),
        )
        t=np.copy(image)
        # for box in bboxes:
        #     t = cv2.rectangle(t, (box[0],box[1]),(box[2],box[3]) , 0, 2)
        #     print(classes[box[4]])
        # cv2.imwrite("test.jpg",t*255)
        return image, bboxes

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device("/cpu:0"):
            self.train_input_size = cfg.TRAIN.INPUT_SIZE
            self.train_output_sizes = self.train_input_size//self.strides
            batch_data = np.zeros(
                (
                    self.batch_size,
                    self.train_input_size,
                    self.train_input_size,
                    3
                ),
                dtype=np.float32
            ) 
            batch_label_small_bounding_box = np.zeros(
                (
                    self.batch_size,
                    self.train_output_sizes[0],
                    self.train_output_sizes[0],
                    self.anchor_per_scale,
                    5+self.num_classes
                ),
                dtype=np.float32
            )
            batch_label_medium_bounding_box = np.zeros(
                (
                    self.batch_size,
                    self.train_output_sizes[0],
                    self.train_output_sizes[0],
                    self.anchor_per_scale,
                    5+self.num_classes
                ),
                dtype=np.float32
            )
            batch_label_large_bounding_box = np.zeros(
                (
                    self.batch_size,
                    self.train_output_sizes[1],
                    self.train_output_sizes[1],
                    self.anchor_per_scale,
                    5+self.num_classes
                ),
                dtype=np.float32
            )
            batch_small_bounding_box=np.zeros(
                (self.batch_size,self.max_bbox_per_scale,4),dtype=np.float32
            )
            batch_medium_bounding_box=np.zeros(
                (self.batch_size,self.max_bbox_per_scale,4),dtype=np.float32
            )
            batch_large_bounding_box=np.zeros(
                (self.batch_size,self.max_bbox_per_scale,4),dtype=np.float32
            )
            num=0
            if self.batch_count<self.num_batchs:
                while num<self.batch_size:
                    index = self.batch_count*self.batch_size + num
                    if index >= self.num_samples:
                        index-=self.num_samples
                    annotation = self.annotations[index]
                    image ,bounding_boxs = self.parse_annotation(annotation)
                    (
                        label_med_bounding_box,
                        label_large_bounding_box,
                        med_bounding_box,
                        large_bound_box
                    ) = self.preprocess_true_boxes(bounding_boxs)
                    batch_data[num,:,:,:]=image
                    #batch_label_small_bounding_box[num,:,:,:] = label_small_bounding_box
                    batch_label_medium_bounding_box[num,:,:,:] = label_med_bounding_box
                    batch_label_large_bounding_box[num,:,:,:] = label_large_bounding_box
                    #batch_small_bounding_box[num,:,:]  = small_bounding_box
                    batch_medium_bounding_box[num,:,:] = med_bounding_box
                    batch_large_bounding_box[num,:,:]  = large_bound_box
                    num+=1
                self.batch_count+=1
                #batch_small_data_label  = batch_label_small_bounding_box , batch_small_bounding_box
                batch_medium_data_label = batch_label_medium_bounding_box, batch_medium_bounding_box
                batch_large_data_label  = batch_label_large_bounding_box , batch_large_bounding_box
                test_b = batch_medium_data_label
                t = np.copy(batch_data*255)

                return (
                    batch_data,
                    (
                        #batch_small_data_label,
                        batch_medium_data_label,
                        batch_large_data_label
                    )
                )
            else:
                self.batch_count=0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def random_horizontal_flip(self,image,bounding_boxes):
        if random.random()<0.5:
            _,w,_=image.shape
            #reverses horizontal raws
            image = image[:,::-1,:]
            # calculates (width - Xul) and (width - Xlr) 
            bounding_boxes[:,[0,2]] = w - bounding_boxes[:,[2,0]]
        return image,bounding_boxes
    def random_crop(self,image,bounding_boxes):
        if random.random()<0.5:
            #finds minimum of all upper left Xs and Ys and max of all lower right Xs and Ys
            #to find the maximum allowed crop
            h, w, _ = image.shape
            allowed_crop = np.concatenate(
                [
                    np.min(bounding_boxes[:, 0:2], axis=0),
                    np.max(bounding_boxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )
            allowed_crop_left  = allowed_crop[0]
            allowed_crop_right = w - allowed_crop[2]
            allowed_crop_up    = allowed_crop[1]
            allowed_crop_down  = h - allowed_crop[3]

            crop_xmin = max(
                0, int(allowed_crop[0] - random.uniform(0, allowed_crop_left))
            )
            crop_ymin = max(
                0, int(allowed_crop[1] - random.uniform(0, allowed_crop_up))
            )
            crop_xmax = max(
                w, int(allowed_crop[2] + random.uniform(0, allowed_crop_right))
            )
            crop_ymax = max(
                h, int(allowed_crop[3] + random.uniform(0, allowed_crop_down))
            )
            image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

            bounding_boxes[:, [0, 2]] = bounding_boxes[:, [0, 2]] - crop_xmin
            bounding_boxes[:, [1, 3]] = bounding_boxes[:, [1, 3]] - crop_ymin
        return image, bounding_boxes
    def random_translate(self, image, bounding_boxes):
        h, w, _ = image.shape
        allowed_crop = np.concatenate(
            [
                np.min(bounding_boxes[:, 0:2], axis=0),
                np.max(bounding_boxes[:, 2:4], axis=0),
            ],
            axis=-1,
        )
        allowed_crop_left  = allowed_crop[0]
        allowed_crop_right = w - allowed_crop[2]
        allowed_crop_up    = allowed_crop[1]
        allowed_crop_down  = h - allowed_crop[3]
        #get random translation from allowed range
        tx = random.uniform(-(allowed_crop_left - 1), (allowed_crop_right - 1))
        ty = random.uniform(-(allowed_crop_up - 1), (allowed_crop_down - 1))

        # m is the transformation matrix
        # x^ = x + tx
        # y^ = y +ty
        M = np.array([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h))
        # shifts the bounding boxes by tx and ty
        bounding_boxes[:, [0, 2]] = bounding_boxes[:, [0, 2]] + tx
        bounding_boxes[:, [1, 3]] = bounding_boxes[:, [1, 3]] + ty

        return image, bounding_boxes

    def preprocess_true_boxes(self,bboxes):
        """
        train_output_size =[64,32,8]
        label -> dimansios = 
        [
            [(32*32) * anchors_per_scale * 5+self.num_classes],
            [(16*16) * anchors_per_scale * 5+self.num_classes],
            [(8*8) * anchors_per_scale * 5+self.num_classes]
        ]

        """
        label = [
            np.zeros(
                (
                    self.train_output_sizes[i],
                    self.train_output_sizes[i],
                    self.anchor_per_scale,
                    5+self.num_classes
                )
            )
            for i in range(2)
        ]

        bboxes_xywh = [np.zeros((self.max_bbox_per_scale,4)) for i in range(2)]
        bbox_count = np.zeros((3,))
        for bbox in bboxes:
            #gets bboxes coordinates Xul Yul Xlr Ylr
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]
            onehot = np.zeros(self.num_classes,dtype=np.float)
            onehot[bbox_class_ind]=1.0
            uniform_distribution = np.full(
                self.num_classes, 1.0 / self.num_classes
            )
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
            #transforms bbox from [Xul, Yul, Xlr, Ylr] -> [centerX, centerY, width, height]
            bbox_xywh = np.concatenate(
                [
                    (bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                    bbox_coor[2:] - bbox_coor[:2],
                ],
                axis=-1,
            )
            
            #scales bbox wrt stride
            bbox_xywh_scaled = (1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis])
            iou = []
            exist_positive = False
            for i in range(2):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                #assignes anchors x and y to centerX and centerY of bounding box
                anchors_xywh[:, 0:2] = (
                    np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                )

                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = utils.bbox_iou(
                    bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh
                )
                #print(iou_scale)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1
                    #print("IN")
                    exist_positive = True

            if not exist_positive:
                #print("NOT IN")
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                #print(best_anchor_ind)
                #print(np.array(iou).reshape(-1).shape)
                #print(best_anchor_ind)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(
                    bbox_xywh_scaled[best_detect, 0:2]
                ).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(
                    bbox_count[best_detect] % self.max_bbox_per_scale
                )
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_mbbox, label_lbbox = label
        mbboxes, lbboxes = bboxes_xywh
        return label_mbbox, label_lbbox, mbboxes, lbboxes








