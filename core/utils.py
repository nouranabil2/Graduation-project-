import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
from core.cfg_config import cfg


def read_class_name(class_file_name):
    """
    inputs: class_file_name:the coco.names file path
    returns : and dict of classIDS and names key=classID values = name
    """
    names=dict()
    with open(class_file_name,"r") as data:
        for ID,name in enumerate(data):
            names[ID] = name.strip("\n")
    return names

def image_preprocess(image,target_size,gt_boxes=None):

    th,tw = target_size
    h,w,_=image.shape
    scale = min(tw/w,th/h)
    nw,nh = int(scale*w),int(scale*h)
    image_resized = cv2.resize(image,(nw,nh))
    image_padded = np.full(shape=[th,tw,3],fill_value=128.0)
    dw, dh = (tw - nw) // 2, (th-nh) // 2
    image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_padded = image_padded / 255.

    if gt_boxes is None:
        return image_padded
    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_padded, gt_boxes


def bbox_iou(bboxes1, bboxes2):
    """
    takes a set of bounding boxes and calculates the intersaction over union
    inputs : bboxes1,bboxes2 -> np arrays of
     [[centerX1,centerY1,width1,height1],
     [centerX2,centerY2,width2,height2],
     [centerX3,centerY3,width3,height3]]
     returns :iou

    """
    #calculates area of all bounding boxes
    #width of boxes  = bboxes1[..., 2]
    #height of boxes = bboxes1[..., 3]
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]
    
    #transforms bboxes1 from [[centerX,centerY,width,height]] -> [[XupperLeft,YupperLeft,XloweRright,YlowerRight]]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    #left_up finds the maximum X and Y from a the set of all UpperLeft Xs and Ys
    #right_down finds the min X and Y from a the set of all lowerRight Xs and Ys

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    return iou
def bbox_GenralizedIou(bboxes1, bboxes2):
    """
    takes a set of bounding boxes and calculates the intersaction over union
    inputs : bboxes1,bboxes2 -> np arrays of
     [[centerX1,centerY1,width1,height1],
     [centerX2,centerY2,width2,height2],
     [centerX3,centerY3,width3,height3]]
     returns :Giou

     iou is not enought to measure the loss when the predicted bounding box has no overlap with the ground truth
     Giou provide a way to distingush between a far pred and a close one 

     giou = iou -(|c/(a union b)|/|c|)
     where a and b are the pred bbox and ground truth
     c is the smallest convex hull that encloses the pred and ground truth boxes

     Giou 

    """
    #calculates area of all bounding boxes
    #width of boxes  = bboxes1[..., 2]
    #height of boxes = bboxes1[..., 3]
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]
    
    #transforms bboxes1 from [[centerX,centerY,width,height]] -> [[XupperLeft,YupperLeft,XloweRright,YlowerRight]]

    bboxes1_coor = tf.concat(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    bboxes2_coor = tf.concat(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    #left_up finds the maximum X and Y from a the set of all UpperLeft Xs and Ys
    #right_down finds the min X and Y from a the set of all lowerRight Xs and Ys

    left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = tf.math.divide_no_nan(inter_area, union_area)

    left_up_coor = tf.minimum(bboxes1_coor[...,:2],bboxes2_coor[...,:2])
    right_down_coor = tf.maximum(bboxes1_coor[...,2:],bboxes2_coor[...,2:])

    convex_hull_intersection = tf.maximum(right_down_coor-left_up_coor,0)

    convex_hull_area = convex_hull_intersection[...,0]*convex_hull_intersection[...,1]
    giou = tf.math.divide_no_nan(convex_hull_area-union_area,convex_hull_area)

    return giou







def load_config():
    STRIDES = np.array(cfg.YOLO.STRIDES)
    ANCHORS = np.array(cfg.YOLO.ANCHORS).reshape(3,3,2)
    XYSCALE = cfg.YOLO.XYSCALE
    NUM_CLASS=len(read_class_name(cfg.YOLO.CLASSES))
    return STRIDES,ANCHORS,XYSCALE,NUM_CLASS



def load_freeze_layer():
    return ['conv2d_93', 'conv2d_101', 'conv2d_109']



def freeze_all(model,frozen=True):
    model.trainable = not frozen
    if isinstance(model,tf.keras.Model):
        print('dddd')
        for layer in model.layers:
            freeze_all(layer,frozen)
def unfreeze_all(model,frozen=False):
    model.trainable = not frozen
    if isinstance(model,tf.keras.Model):
        for layer in model.layers:
            unfreeze_all(layer,frozen)


def non_maximum_suppression(bounding_boxes,iou_threshold,sigma=0.3,method="nms"):
    pass

def decode_tf(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]):
    batch_size = tf.shape(conv_output)[0]
    conv_output = tf.reshape(conv_output,
                             (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS),
                                                                          axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
              STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    pred_prob = pred_conf * pred_prob
    pred_prob = tf.reshape(pred_prob, (batch_size, -1, NUM_CLASS))
    pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))

    return pred_xywh, pred_prob
def load_weights(model,weights_path,is_darkNet=False):
    layer_size=110
    output_pos = [93,101,109]
    with open(weights_path,"rb") as f:
        m,n,rev,s,_=np.fromfile(f,dtype=np.int32,count=5)
        for i in range(layer_size):
            conv_name= f"conv2d_{i}" if i>0 else "conv2d"
            bn_name=f"batch_normalization_{i}" if i>0 else "batch_normalization"
            layer = model.get_layer(conv_name)
            filters = layer.filters
            kernel_size=layer.kernel_size[0]
            in_dim =layer.input_shape[-1]
            
            if i not in output_pos:
                batch_normalization_weights = np.fromfile(f,dtype=np.float32,count=4*filters)
                batch_normalization_weights = batch_normalization_weights.reshape((4,filters))[[1,0,2,3]]
                bn_layer = model.get_layer(bn_name)
            else:
                conv_bais = np.fromfile(f,dtype=np.float32,count=filters)
            conv_shape = (filters,in_dim,kernel_size,kernel_size)
            conv_weights = np.fromfile(f,dtype=np.float32,count=np.product(conv_shape))
            conv_weights = conv_weights.reshape(conv_shape).transpose([2,3,1,0])

            if i not in output_pos:
                layer.set_weights([conv_weights])
                bn_layer.set_weights(batch_normalization_weights)
            else:
                layer.set_weights([conv_weights,conv_bais])
        f.close()

def load_darknet(model,weights_path):
    layer_size=71
    with open(weights_path,"rb") as weights_file:

        major, minor, revision = np.fromfile(weights_file,dtype=np.int32,count=3)
        
        if (major*10+minor)>=2 and major<1000 and minor<1000:
            seen = np.fromfile(weights_file,dtype=np.int64,count=1)
        else:
            seen =np.fromfile(weights_file,dtype=np.int32,count=1)
        print(f"MINOR {minor} MAJOR {major} REVISION{revision} ,SEEN{type(seen[0])}{seen}")

        for i in range(layer_size+1):
            conv_name= f"conv2d_{i}" if i>0 else "conv2d"
            bn_name=f"batch_normalization_{i}" if i>0 else "batch_normalization"
            #print(F"=====================LOADING CONVOLUOTIONAL i = {i+1}===========================")
            layer = model.get_layer(conv_name)
            filters = layer.filters
            kernel_size=layer.kernel_size[0]
            in_dim =layer.input_shape[-1]
            s = "filters:"+str(filters)
            
            if i != 72:
                s+=" , bn "
                batch_normalization_weights = np.fromfile(weights_file,dtype=np.float32,count=4*filters)
                batch_normalization_weights = batch_normalization_weights.reshape((4,filters))[[1,0,2,3]]
                names = ["scales","bais","mean" "variance"]
                #print(batch_normalization_weights)
                bn_layer = model.get_layer(bn_name)
            else:
                conv_bias = np.fromfile(weights_file, dtype=np.float32, count=filters)

            conv_shape = (filters,in_dim,kernel_size,kernel_size)
            conv_weights = np.fromfile(weights_file,dtype=np.float32,count=np.product(conv_shape))
            #print(conv_weights)
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])
            

            if i !=72:
                layer.set_weights([conv_weights])
                bn_layer.set_weights(batch_normalization_weights)
            else:
                layer.set_weights([conv_weights,conv_bias])
        weights_file.close()