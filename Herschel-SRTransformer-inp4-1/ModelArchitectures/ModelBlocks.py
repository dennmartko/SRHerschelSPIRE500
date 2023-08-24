######################
###     IMPORTS    ###
######################
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Input, Dense, LeakyReLU, Conv2DTranspose, LayerNormalization

class SpatialSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_channels, name=None):
        super(SpatialSelfAttention, self).__init__(name=name)

        # Define linear transformations for queries, keys, and values
        self.num_channels = num_channels

        self.query_convs = tf.keras.layers.Conv2D(num_channels, kernel_size=1, data_format="channels_last", strides=1)
        self.key_convs = tf.keras.layers.Conv2D(num_channels, kernel_size=1, data_format="channels_last", strides=1)
        self.value_convs = tf.keras.layers.Conv2D(num_channels, kernel_size=1, data_format="channels_last", strides=1)

        #self.final_conv = tf.keras.layers.Conv2D(num_channels, kernel_size=1, data_format="channels_first", strides=1)

        # Define scaling factor for dot product
        self.scale_factor = tf.math.sqrt(tf.cast(num_channels, dtype=tf.float32))
        
    # 1 head
    def split_heads(self, x):
        x = tf.keras.layers.Reshape((x.shape[2], x.shape[3], self.num_channels))(x)
        return x

    def call(self, inputs):
        x = self.split_heads(inputs)

        resh = tf.keras.layers.Reshape((inputs.shape[2] * inputs.shape[3], self.num_channels))
        resh_final = tf.keras.layers.Reshape((self.num_channels, inputs.shape[2], inputs.shape[3]))

        q = self.query_convs(x)
        k = self.key_convs(x)
        v = self.value_convs(x)

        # dot-product attention
        attention = tf.matmul(resh(q), resh(k), transpose_b=True) / self.scale_factor
        attention = tf.keras.activations.softmax(attention)

        # Calculate attention output
        output = tf.matmul(attention, resh(v))
        output = resh_final(output)

        # Final convolution to concatenated outputs
        #outputs = self.final_conv(outputs)
        return output

class ChannelSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_channels, name=None):
        super(ChannelSelfAttention, self).__init__(name=name)

        # Define linear transformations for queries, keys, and values
        self.num_channels = num_channels

        self.query_convs = tf.keras.layers.Conv2D(num_channels, kernel_size=1, data_format="channels_first", strides=1)
        self.key_convs = tf.keras.layers.Conv2D(num_channels, kernel_size=1, data_format="channels_first", strides=1)
        self.value_convs = tf.keras.layers.Conv2D(num_channels, kernel_size=1, data_format="channels_first", strides=1)

        # Define scaling factor for dot product
        self.scale_factor = tf.math.sqrt(tf.cast(num_channels, dtype=tf.float32))

    def call(self, inputs):
        # Reshape for each head
        resh = tf.keras.layers.Reshape((self.num_channels, inputs.shape[2] * inputs.shape[3]))
        # Final reshape layer
        out_resh = tf.keras.layers.Reshape((self.num_channels, inputs.shape[2], inputs.shape[3]))
        q = self.query_convs(inputs)
        k = self.key_convs(inputs)
        v = self.value_convs(inputs)

        # dot-product attention
        attention = tf.matmul(resh(q), resh(k), transpose_b=True) / self.scale_factor
        attention = tf.keras.activations.softmax(attention)

        # Calculate attention output
        output = tf.matmul(attention, resh(v))
        output = out_resh(output)

        return output

def PCAM(conv_params_out, inp, axis, ID):
    # First Attention block
    ## Scaling parameters
    a = tf.Variable(initial_value=0., dtype=tf.float32, trainable=True, name=f"a_{ID}")
    b = tf.Variable(initial_value=0., dtype=tf.float32, trainable=True, name=f"b_{ID}")

    ## Spatial Attention
    spat_att = SpatialSelfAttention(num_channels=inp.shape[axis], name=f"MHSA_{ID}")(inp)
    conv_spat = Conv2D(filters=inp.shape[axis], kernel_size=(1,1), strides=(1,1), padding='same', data_format=conv_params_out["data_format"])(spat_att)
    ## Channel Attention
    chan_att = ChannelSelfAttention(num_channels=inp.shape[axis], name=f"CSA_{ID}")(inp)
    conv_chan = Conv2D(filters=inp.shape[axis], kernel_size=(1,1), strides=(1,1), padding='same', data_format=conv_params_out["data_format"])(chan_att)  

    ## Fusion
    fusion_spat = tf.keras.layers.Add(name=f"SpatFusion_{ID}")([b*conv_spat, inp])
    fusion_chan = tf.keras.layers.Add(name=f"ChanFusion_{ID}")([a*conv_chan, inp])
    fusion = tf.keras.layers.Add(name=f"Fusion_{ID}")([fusion_spat, fusion_chan])
    # sam_out = Conv2D(filters=conv_params_out["filters"], kernel_size=(1,1), strides=(1,1), padding='same', data_format=conv_params_out["data_format"])(fusion)
    #sam_out = tf.keras.activations.swish(sam_out)
    return fusion

def CAM(conv_params_out, inp, axis, ID):
    # First Attention block
    ## Scaling parameters
    a = tf.Variable(initial_value=0., dtype=tf.float32, trainable=True, name=f"a_{ID}")

    ## Channel Attention
    chan_att = ChannelSelfAttention(num_channels=inp.shape[axis], name=f"CSA_{ID}")(inp)
    conv_fusion = Conv2D(filters=inp.shape[axis], kernel_size=(1,1), strides=(1,1), padding='same', data_format=conv_params_out["data_format"])(chan_att)

    ## Fusion
    fusion = tf.keras.layers.Add(name=f"ChanFusion_{ID}")([a*conv_fusion, inp])
    return fusion

def ConvMultiScaleBlock(conv_params_in, conv_params_out, bn_params, drop_params, inp, axis, ID):
    # Multi-scale block  
    ## Initial high-level feature extraction
    x = Conv2D(**conv_params_in, kernel_size=(5, 5), strides=(1,1))(inp)
    x = BatchNormalization(**bn_params)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = Conv2D(**conv_params_in, kernel_size=(3, 3), strides=(1,1))(x)
    x = BatchNormalization(**bn_params)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    if x.shape[2] == 106:
        X = tf.keras.layers.Concatenate(axis=axis)([x, inp])
    else:
        X = tf.keras.layers.Add()([x, inp])

    # if X.shape[axis] % num_heads != 0:
    #     X = Conv2D(**conv_params_in, kernel_size=(1, 1), strides=(1,1))(X)
    if inp.shape[2] >= 106:
        sam_out = CAM(conv_params_in, X, axis, ID)
    else:
        sam_out = PCAM(conv_params_in, X, axis, ID)

    # DownSample 4x4 
    out = Conv2D(**conv_params_out, kernel_size=(4, 4), strides=(2,2))(x)
    out = BatchNormalization(**bn_params)(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.2)(out)
    out = tf.keras.layers.Dropout(**drop_params)(out)
    return out, sam_out


def DeConvMultiScaleBlock(conv_params, deconv_params, bn_params, drop_params, inp, axis, residual = None, extra_skip = None):
    # SAM connects skip connection
    ## If we can not divide the channels into h number of heads, then we need to scale the filters

    # Multi-scale block  
    ## Initial high-level feature extraction
    x = Conv2D(**conv_params, kernel_size=(5, 5), strides=(1,1))(inp)
    x = BatchNormalization(**bn_params)(x)
    #x = tf.keras.activations.swish(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = Conv2D(**conv_params, kernel_size=(3, 3), strides=(1,1))(x)
    x = BatchNormalization(**bn_params)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    if residual is not None:
        x = tf.keras.layers.Add()([x, residual])

    if extra_skip is not None:
        x = tf.keras.layers.Concatenate(axis=axis)([x, extra_skip])

    # 4x4 
    out = Conv2DTranspose(**deconv_params, kernel_size=(4, 4), strides=(2,2))(x)
    out = BatchNormalization(**bn_params)(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.2)(out)
    out = tf.keras.layers.Dropout(**drop_params)(out)
    return out

def BottleneckBlock(conv_params, deconv_params, bn_params, drop_params, inp, axis):
    x = Conv2D(**conv_params, kernel_size=(3, 3), strides=(1,1))(inp)
    x = BatchNormalization(**bn_params)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(**conv_params, kernel_size=(3, 3), strides=(1,1))(x)
    x = BatchNormalization(**bn_params)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    x = tf.keras.layers.Add()([x, inp])
    # 4x4 
    out = Conv2DTranspose(**deconv_params, kernel_size=(4, 4), strides=(2,2))(x)
    out = BatchNormalization(**bn_params)(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.2)(out)
    out = tf.keras.layers.Dropout(**drop_params)(out)
    return out

def ConvolutionBlock(conv_params, bn_params, act_params, drop_params, inp, use_bn, use_ln=False):
    x = Conv2D(**conv_params)(inp)
    if use_bn:
        x = BatchNormalization(**bn_params)(x)
    if use_ln:
        x = LayerNormalization(epsilon=1e-4)(x)
    #x = LeakyReLU(**act_params)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = tf.keras.layers.Dropout(**drop_params)(x)
    return x