import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.DataSet import Dataset
from core.cfg_config import cfg
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
testData  = Dataset(is_training=False)
trainData  = Dataset(is_training=True)
path="checkpoints/check_10-4/"
print("loading weights")
name = 'yoloV4_19'
saved_model_loaded = tf.saved_model.load(path+name, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']
for image_data ,target in trainData:
    batch_data = tf.constant(image_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox(image_data[0]*255, pred_bbox)
        # image = utils.draw_bbox(image_data*255, pred_bbox)
        image = Image.fromarray(image.astype(np.uint8))
        image.show()