# import tensorflow as tf
# import core.block as block

from lib2to3.pytree import convert
from unicodedata import name
import tensorflow as tf
import core.block as block
from core.utils import *



import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.python.keras.utils import*

def cspdarknet53(input_data):
    #(kernel_width,kernel_height,input_channel,output_channel)
    input_data = block.convolutional_block(input_data,(3,3,3,32))#
    input_data = block.convolutional_block(input_data,(3,3,32,64),downsample=True)#
    route = input_data

    route = block.convolutional_block(route,(1,1,64,64))#
    input_data = block.convolutional_block(input_data,(1,1,64,64))
    #resiudual block(input_data,input_channel,filter_number1,filter_number2)
    #return s input+conv
    #conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
    #conv = convolutional(conv       , filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)
    input_data = block.residual_block(input_data,64,32,64)
    input_data = block.convolutional_block(input_data,(1,1,64,64))

    input_data = tf.concat([input_data,route],axis=-1)

    input_data = block.convolutional_block(input_data,(1,1,128,64))
    input_data = block.convolutional_block(input_data,(3,3,64,128),downsample=True)
    route = input_data
    route = block.convolutional_block(route,(1,1,128,64))
    input_data = block.convolutional_block(input_data,(1,1,128,64))

    for i in range(2):
        input_data = block.residual_block(input_data,64,64,64)
    input_data = block.convolutional_block(input_data,(1,1,64,64))
    input_data = tf.concat([input_data,route],axis=-1)

    input_data = block.convolutional_block(input_data,(1,1,128,128))
    input_data = block.convolutional_block(input_data,(3,3,128,256),downsample=True)
    route = input_data
    route = block.convolutional_block(route,(1,1,256,128))
    input_data = block.convolutional_block(input_data,(1,1,256,128))

    for i in range(8):
        input_data = block.residual_block(input_data,128,128,128)
    input_data = block.convolutional_block(input_data,(1,1,128,128))
    input_data = tf.concat([input_data,route],axis=-1)

    input_data = block.convolutional_block(input_data,(1,1,256,256))
    first_route = input_data

    input_data = block.convolutional_block(input_data,(3,3,256,512),downsample=True)
    route = input_data
    route=block.convolutional_block(route,(1,1,512,256))
    input_data=block.convolutional_block(input_data,(1,1,512,256))

    for i in range(8):
        input_data=block.residual_block(input_data,256,256,256)
    input_data=block.convolutional_block(input_data,(1,1,256,256))
    input_data = tf.concat([input_data,route],axis=-1)
    input_data = block.convolutional_block(input_data,(1,1,512,512))
    second_route = input_data
    input_data = block.convolutional_block(input_data,(3,3,512,1024),downsample=True)
    route=input_data
    route = block.convolutional_block(route,(1,1,1024,512))
    input_data = block.convolutional_block(input_data,(1,1,1024,512))

    for i in range(4):
        input_data = block.residual_block(input_data,512,512,512)
    input_data = block.convolutional_block(input_data,(1,1,512,512))
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = block.convolutional_block(input_data,(1,1,1024,1024))

    return first_route,second_route,input_data

# tf.config.run_functions_eagerly(True)
# input_layer = tf.keras.layers.Input([256,256, 3])
# _1,_2,yolo = cspdarknet53(input_layer)
# yolo = tf.keras.layers.AveragePooling2D(pool_size=(8,8))(yolo)
# yolo = tf.keras.layers.Conv2D(filters=1000,kernel_size=1,strides=1,padding="same") (yolo)
# yolo = tf.keras.layers.Softmax()(yolo)
# model = tf.keras.Model(inputs=input_layer, outputs=yolo)

#model.load_weights('./cspdarknet53.h5')
#model = tf.keras.models.load_model('cspdarknet53.h5',custom_objects={'mish': mish})
# load_darknet(model,"../csdarknet53-omega_final.weights")
#model.summary()
# print(np.array(model.get_layer("conv2d_72").get_weights()[1]).shape)





# filename = '../data/eagle.jpg'


# from PIL import Image

# image = Image.open(filename).resize((256,256),resample=Image.BILINEAR)

# image_array = np.array(image,dtype=np.float32)/255.0

# image_array=image_array[np.newaxis,...]
# x = tf.keras.Input(shape=(256,256))
# x= tf.convert_to_tensor(image_array)

# predictions_dark =model(x)# model.predict(image_array)

# predictions_vgg = vgg_model.predict(processed_image)
# label_vgg = decode_predictions(tf.reshape(predictions_vgg,(1,1000)).numpy())
# for prediction_id in range(len(label_vgg[0])):
#     print(label_vgg[0][prediction_id])

# print("++++++++++++++++++++++++++++===")

# dark = np.argmax(tf.reshape(predictions_dark,(1,1000)).numpy())
# print(dark)
# predictions_vgg = model.predict(x)
# label_vgg = decode_predictions(tf.reshape(predictions_vgg,(1,1000)).numpy())
# for prediction_id in range(len(label_vgg[0])):
#   print(label_vgg[0][prediction_id])


