from posixpath import split
import tensorflow as tf
class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self,x,training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training,self.trainable)
        return super().call(x,training)

def convolutional_block(input_layer, filter_shape, activate=True, batch_normalize=True, downsample=False, activation_func="mish"):
    if downsample:
        input_layer=tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))(input_layer)
        padding ="valid"
        strides = 2
    else:
        strides = 1
        padding = "same"
    #filter_shape[-1] -> dimension of output space
    conv = tf.keras.layers.Conv2D(filters=filter_shape[-1],kernel_size=filter_shape[0],strides=strides,padding=padding,use_bias=not batch_normalize,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)
    if batch_normalize:
        conv = BatchNormalization()(conv)
    if activate:
        if activation_func=="mish":
            conv = mish(conv)
        elif activation_func=="leaky_relu":
            conv = tf.nn.leaky_relu(conv,alpha=0.1)
    return conv
    
def mish (layer):
    return layer*tf.math.tanh(tf.math.softplus(layer))

def residual_block(input_layer,input_channel,filter1,filter2,activation_func="mish"):
    pass_route = input_layer
    conv = convolutional_block(input_layer,(1,1,input_channel,filter1),activation_func=activation_func)
    conv = convolutional_block(conv,(3,3,filter1,filter2),activation_func=activation_func)
    res_block_output = pass_route+conv
    return res_block_output
def upsample(layer):
    return tf.image.resize(layer,(layer.shape[1]*2,layer.shape[2]*2),method="bilinear")
def route_group(input_layer,groups,group_id):
    sub_layers = tf.split(input_layer,num_or_size_splits=groups,axis=-1)
    return sub_layers[group_id]