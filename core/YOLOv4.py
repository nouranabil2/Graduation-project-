from turtle import shape
import numpy as np
import tensorflow as tf
import utils as utils
import block as block
import CSPdarknet53 as CSPdarknet53
from cfg_config import cfg

def YOLOv4(input_layer, NUM_CLASS):
    first_route,second_route,conv = CSPdarknet53.cspdarknet53(input_layer)

    route = conv 
    conv = block.convolutional_block(conv,(1,1,512,256),activation_func="leaky_relu")
    conv = block.upsample(conv)

    second_route = block.convolutional_block(second_route,(1,1,512,256),activation_func="leaky_relu")

    conv = tf.concat([second_route,conv],axis = -1)

    conv = block.convolutional_block(conv,(1,1,512,256),activation_func="leaky_relu")
    conv = block.convolutional_block(conv,(3,3,256,512),activation_func="leaky_relu")
    conv = block.convolutional_block(conv,(1,1,512,256),activation_func="leaky_relu")
    conv = block.convolutional_block(conv,(3,3,256,512),activation_func="leaky_relu")
    conv = block.convolutional_block(conv,(1,1,512,256),activation_func="leaky_relu")

    second_route = conv

    conv = block.convolutional_block(conv,(1,1,256,128),activation_func="leaky_relu")
    conv = block.upsample(conv)

    first_route = block.convolutional_block(first_route,(1,1,256,128),activation_func="leaky_relu")
    conv = tf.concat([first_route,conv],axis=-1)

    conv = block.convolutional_block(conv,(1,1,256,128),activation_func="leaky_relu")
    conv = block.convolutional_block(conv,(3,3,128,256),activation_func="leaky_relu")
    conv = block.convolutional_block(conv,(1,1,256,128),activation_func="leaky_relu")
    conv = block.convolutional_block(conv,(3,3,128,256),activation_func="leaky_relu")
    conv = block.convolutional_block(conv,(1,1,256,128),activation_func="leaky_relu")

    first_route = conv 
    conv = block.convolutional_block(conv,(3,3,128,256),activation_func="leaky_relu")

    small_bounding_box = block.convolutional_block(conv,(1,1,256,3*(NUM_CLASS+5)),activate=False,batch_normalize=False,activation_func="leaky_relu")

    conv = block.convolutional_block(first_route,(3,3,128,256),downsample=True,activation_func="leaky_relu")
    conv=tf.concat([conv,second_route],axis=-1)

    conv = block.convolutional_block(conv,(1,1,512,256),activation_func="leaky_relu")
    conv = block.convolutional_block(conv,(3,3,256,512),activation_func="leaky_relu")
    conv = block.convolutional_block(conv,(1,1,512,256),activation_func="leaky_relu")
    conv = block.convolutional_block(conv,(3,3,256,512),activation_func="leaky_relu")
    conv = block.convolutional_block(conv,(1,1,512,256),activation_func="leaky_relu")

    second_route =conv
    conv = block.convolutional_block(conv,(3,3,256,512),activation_func="leaky_relu")
    med_bounding_box = block.convolutional_block(conv,(1,1,512,3*(NUM_CLASS+5)),activate=False,batch_normalize=False,activation_func="leaky_relu")

    conv = block.convolutional_block(second_route,(3,3,256,512),downsample=True,activation_func="leaky_relu")
    conv=tf.concat([conv,route],axis=-1)

    conv = block.convolutional_block(conv,(1,1,1024,512),activation_func="leaky_relu")
    conv = block.convolutional_block(conv,(3,3,512,1024),activation_func="leaky_relu")
    conv = block.convolutional_block(conv,(1,1,1024,512),activation_func="leaky_relu")
    conv = block.convolutional_block(conv,(3,3,512,1024),activation_func="leaky_relu")
    conv = block.convolutional_block(conv,(1,1,1024,512),activation_func="leaky_relu")

    conv = block.convolutional_block(conv,(3,3,512,1024),activation_func="leaky_relu")
    large_bounding_box = block.convolutional_block(conv,(1,1,1024,3*(NUM_CLASS+5)),activate=False,batch_normalize=False,activation_func="leaky_relu")
    return[small_bounding_box,med_bounding_box,large_bounding_box]

# input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
# _1,_2,yolo = YOLOv4(input_layer,80)
# model = tf.keras.Model(inputs=input_layer, outputs=yolo)
# model.summary()
def decode_output(net_output,output_size,NUM_CLASS,STRIDES,ANCHORS,i,XYSCALE=[1,1,1]):
    net_output = tf.reshape(net_output,(tf.shape(net_output[0],output_size,output_size,3,5+NUM_CLASS) ))
    dxdy,dwdh,confidance,probabilty = tf.split(net_output,(2,2,1,NUM_CLASS),axis=-1)
    # meshgrid returns to arrays 
    #one for x values and one for y values 
    """
    example :
    meshgrid(tf.range(3),tf.range(3))


    [<tf.Tensor: shape=(3, 3), dtype=int32, numpy=
    array([[0, 1, 2],
           [0, 1, 2],
           [0, 1, 2]], dtype=int32)>, <tf.Tensor: shape=(3, 3), dtype=int32, numpy=
    array([[0, 0, 0],
           [1, 1, 1],
           [2, 2, 2]], dtype=int32)>]

    then expand_dims returns 
    (3,3,1,2) array after stacking on axis -1
    then 
    then expand_dim axis 0
    (1,3,3,1,2)
    then tile repeats the elements of axis_T Nt times
    for tile(array,[n0,n1,n2,....nT])
    example [64,1,1,3,1]
    the output 
    (64,3,3,3,2)
    genral case
    (conv_output.shape[0],output_size,output_size,number_of_anchors,2)



    """
    xy_grid = tf.meshgrid(tf.range(output_size),tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid,axis=-1),axis=2)
    xy_grid = tf.tile(tf.expand_dims(xy_grid,axis=0),[tf.shape(net_output)[0],1,1,3,1])

    xy_grid = tf.cast(xy_grid,tf.float32)

    pred_xy = ((tf.sigmoid(dxdy)*XYSCALE[i]) - 0.5*(XYSCALE[i]-1)+xy_grid)*STRIDES[i]
    pred_wh = (tf.exp(dwdh))*ANCHORS[i]
    xywh = tf.concat([pred_xy,pred_wh],axis=-1)
    pred_conf=tf.sigmoid(confidance)
    pred_prob = tf.sigmoid(probabilty)
    return tf.concat([xywh,pred_conf,pred_prob],axis=-1)
def compute_loss(predication,conv,label,bounding_boxes,STRIDES,IOU_LOSS_THRESH,NUM_CLASS,i=0):
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = STRIDES[i]*output_size
    conv=tf.reshape(conv,(batch_size,output_size,output_size,3,5+NUM_CLASS))

    raw_conf = conv[:,:,:,:,4:5]
    raw_prob = conv[:,:,:,:,5:]
    predection_xywh = predication[:,:,:,:,0:4]
    predection_conf = predication[:,:,:,:,4:5]

    label_xywh = label[:,:,:,:,0:4]
    obj = label[:,:,:,:,4:5]
    label_prob = label[:,:,:,:,5:]

    giou = tf.expand_dims(utils.bbox_GenralizedIou(predection_xywh,label_xywh),axis=-1)
    input_size = tf.cast(input_size,tf.float32)
    bounding_boxes_loss_scale = 2.0 - 1.0*(label_xywh[:,:,:,:2:3]*label_xywh[:,:,:,:,3:4])/(input_size**2)
    giou_loss = obj * bounding_boxes_loss_scale *(1-giou)

    iou = utils.bbox_iou(predection_xywh[:,:,:,:,np.newaxis,:],bounding_boxes[:,np.newaxis,:np.newaxis,np.newaxis,:,:])
    max_iou = tf.expand_dims(tf.reduce_max(iou,axis=-1),axis=-1)
    noobj = (1-obj) * tf.cast(max_iou>IOU_LOSS_THRESH,tf.float32)
    confidance = tf.pow(obj-predection_conf,2)
    obj_sigmoid_loss = obj * tf.nn.sigmoid_cross_entropy_with_logits(labels=obj,logits=raw_conf)
    noobj_sigmoid_loss = noobj * tf.nn.sigmoid_cross_entropy_with_logits(labels=noobj,logits=raw_conf)
    confidance_loss = confidance *(obj_sigmoid_loss + noobj_sigmoid_loss)
    prob_loss = obj * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob,logits=raw_prob)


    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss,axis=[1,2,3,4]))
    confidance_loss = tf.reduce_mean(tf.reduce_sum(confidance_loss,axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss,axis=[1,2,3,4]))

    return giou_loss,confidance_loss,prob_loss






















    