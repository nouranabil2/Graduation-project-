import tensorflow as tf
import block as block



def cspdarknet53(input_data):
    #(kernel_width,kernel_height,input_channel,output_channel)
    input_data = block.convolutional_block(input_data,(3,3,3,32))
    input_data = block.convolutional_block(input_data,(3,3,32,64),downsample=True)
    route = input_data

    route = block.convolutional_block(route,(1,1,64,64))
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
    input_data = block.convolutional_block(input_data,(1,1,1024,512),activation_func="leaky_relu")
    input_data = block.convolutional_block(input_data,(3,3,512,1024),activation_func="leaky_relu")
    input_data = block.convolutional_block(input_data,(1,1,1024,512),activation_func="leaky_relu")

    input_data = tf.concat([tf.nn.max_pool(input_data, ksize=13, padding='SAME', strides=1), tf.nn.max_pool(input_data, ksize=9, padding='SAME', strides=1)
                            , tf.nn.max_pool(input_data, ksize=5, padding='SAME', strides=1), input_data], axis=-1)
    input_data = block.convolutional_block(input_data,(1,1,2048,512),activation_func="leaky_relu")
    input_data = block.convolutional_block(input_data,(3,3,512,1024),activation_func="leaky_relu")
    input_data = block.convolutional_block(input_data,(1,1,1024,512),activation_func="leaky_relu")
    return first_route,second_route,input_data
    
input_layer = tf.keras.layers.Input([512,512, 3])
_1,_2,yolo = cspdarknet53(input_layer)
model = tf.keras.Model(inputs=input_layer, outputs=yolo)
model.summary()













    

