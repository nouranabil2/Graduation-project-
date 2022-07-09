import os 
import shutil
import tensorflow as tf
from core.YOLOv4 import YOLOv4 ,decode_output,compute_loss,YOLOv4_tiny
from core.DataSet import Dataset
from core.cfg_config import cfg
from core import CSPdarknet53
import numpy as np
from core import utils

WEIGHTS_PATH = "./yolov4.weights"
LOG_DIR = "./data/log"
def train_step(image_data,target):
    with tf.GradientTape() as tape:
        predications = model(image_data,training=True)
        giou_loss = 0
        conf_loss = 0
        prob_loss = 0
        for i in range(len(freeze_layers)):
            net,pred = predications[i*2],predications[i*2+1]
            loss = compute_loss(pred,net,target[i][0],target[i][1],STRIDES=STRIDES,NUM_CLASS=NUM_CLASS,IOU_LOSS_THRESH=IOU_LOSS_THRESH,i=i)
            giou_loss += loss[0]
            conf_loss += loss[1]
            prob_loss += loss[2]
        total_loss = giou_loss + conf_loss + prob_loss
        grad = tape.gradient(total_loss,model.trainable_variables)
        opt.apply_gradients(zip(grad,model.trainable_variables))
        tf.print("=> STEP %4d/%4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, total_steps, opt.lr.numpy(),
                                                               giou_loss, conf_loss,
                                                               prob_loss, total_loss))
        global_steps.assign_add(1)
        if global_steps<warmup_steps:
            lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
        else:
            lr = (cfg.TRAIN.LR_END) +0.5*((cfg.TRAIN.LR_INIT)-(cfg.TRAIN.LR_END))*(
                (1+tf.cos((global_steps-warmup_steps)/(total_steps-warmup_steps)*np.pi))
            )
        opt.lr.assign(lr.numpy())
        with writer.as_default():
                tf.summary.scalar("lr", opt.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
        writer.flush()
def test_step(image_data,target):
    predications = model(image_data,training=False)
    giou_loss = 0
    conf_loss = 0
    prob_loss = 0
    for i in range(len(freeze_layers)):
        net,pred = predications[i*2],predications[i*2+1]
        loss = compute_loss(pred,net,target[i][0],target[i][1],STRIDES=STRIDES,NUM_CLASS=NUM_CLASS,IOU_LOSS_THRESH=IOU_LOSS_THRESH,i=i)
        giou_loss += loss[0]
        conf_loss += loss[1]
        prob_loss += loss[2]
    total_loss = giou_loss + conf_loss + prob_loss
    global_test_step.assign_add(1)
    tf.print("=> TEST STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
                    "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, giou_loss, conf_loss,
                                                            prob_loss, total_loss))
    with writer.as_default():
                tf.summary.scalar("test_loss/total_loss", total_loss, step=global_test_step)
                tf.summary.scalar("test_loss/giou_loss", giou_loss, step=global_test_step)
                tf.summary.scalar("test_loss/conf_loss", conf_loss, step=global_test_step)
                tf.summary.scalar("test_loss/prob_loss", prob_loss, step=global_test_step)
    writer.flush()


if __name__=="__main__":
    #def_gpu = tf.config.experimental.list_physical_devices("GPU")
    # if len(def_gpu)>0:
    #     tf.config.experimental.set_memory_growth(def_gpu[0],True)
    trainData = Dataset(is_training=True)
    testData  = Dataset(is_training=False)
    isfreeze=False
    steps_per_epoch = len(trainData)
    first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
    second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
    global_steps = tf.Variable(1,trainable=False,dtype=tf.int64)
    global_test_step = tf.Variable(1,trainable=False,dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps= (first_stage_epochs+second_stage_epochs)*steps_per_epoch
    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE,cfg.TRAIN.INPUT_SIZE,3])

    STRIDES ,ANCHORS,XYSCALE,NUM_CLASS = utils.load_config(tiny=True)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
    freeze_layers = utils.load_freeze_layer(tiny=True)
    feature_map = YOLOv4_tiny(input_layer,NUM_CLASS)
    bounding_boxes =[]
    for i ,feature in enumerate(feature_map):
        if i == 0:
            bounding_box = decode_output(feature,cfg.TRAIN.INPUT_SIZE//16,NUM_CLASS,STRIDES,ANCHORS,i,XYSCALE)
        else:
            bounding_box = decode_output(feature,cfg.TRAIN.INPUT_SIZE//32,NUM_CLASS,STRIDES,ANCHORS,i,XYSCALE)
        bounding_boxes.append(feature)
        bounding_boxes.append(bounding_box)
    model = tf.keras.Model(input_layer,bounding_boxes)
    #utils.load_darknet(model,"./csdarknet53-omega_final.weights")
    #model = tf.saved_model.load('saved_model')
    model.summary()
    path="checkpoints/best_yet/"
    print("loading weights")
    name = 'yoloV4_29'
    model.load_weights(path+name)
    
    opt = tf.keras.optimizers.Adam()
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    writer=tf.summary.create_file_writer(LOG_DIR)
    for epoch in range(first_stage_epochs+second_stage_epochs):
        if epoch < first_stage_epochs:
            if not isfreeze:
                isfreeze=True
                for name in freeze_layers:
                    freeze = model.get_layer(name)
                    utils.freeze_all(freeze)
        elif epoch >= first_stage_epochs:
            if isfreeze:
                isfreeze=False
                for name in freeze_layers:
                    freeze = model.get_layer(name)
                    utils.unfreez_all(freeze)
        for image_data ,target in trainData:
            train_step(image_data,target)
        for image_data ,target in testData:
            test_step(image_data,target)
        model.save_weights(f"./checkpoints/yoloV4_{epoch}")
        




