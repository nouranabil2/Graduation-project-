import tensorflow as tf
from core.YOLOv4 import YOLOv4, decode_tf, filter_boxes
import core.utils as utils
from core.cfg_config import cfg

WEIGHTS_PATH ='./yolov4-csp.weights'
OUTPUT_PATH="./data/yolov4-416"
INPUT_SIZE = cfg.INPUT_SIZE

def save_tf():
  STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

  input_layer = tf.keras.layers.Input([INPUT_SIZE,INPUT_SIZE, 3])
  feature_maps = YOLOv4(input_layer, NUM_CLASS)
  bbox_tensors = []
  prob_tensors = []
  for i, fm in enumerate(feature_maps):
    if i == 0:
      output_tensors = decode_tf(fm, INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
    elif i == 1:
      output_tensors = decode_tf(fm, INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
    else:
      output_tensors = decode_tf(fm,INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
    bbox_tensors.append(output_tensors[0])
    prob_tensors.append(output_tensors[1])
  pred_bbox = tf.concat(bbox_tensors, axis=1)
  pred_prob = tf.concat(prob_tensors, axis=1)

  boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=FLAGS.score_thres, input_shape=tf.constant([FLAGS.input_size, FLAGS.input_size]))
  pred = tf.concat([boxes, pred_conf], axis=-1)
  model = tf.keras.Model(input_layer, pred)

  utils.load_weights(model, WEIGHTS_PATH)
  model.summary()
  model.save(OUTPUT_PATH)
  
def main(_argv):
  save_tf()
main()