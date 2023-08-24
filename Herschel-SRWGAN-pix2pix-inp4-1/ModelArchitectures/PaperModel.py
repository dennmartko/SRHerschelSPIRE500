######################
###     IMPORTS    ###
######################
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Input, Dense, LeakyReLU, Flatten, Conv2DTranspose, Dropout
from ModelArchitectures.ModelBlocks import ConvolutionBlock, DenseBlock

def Generator(shape, data_format, C1, K, multipliers):
    axis = 1 if shape[0] != shape[1] else 3
    # Block parameters
    shape = shape
    conv_params = lambda n, regularize_bool: {'filters':n*C1, 'use_bias':True, 'padding':'same', 'data_format': data_format, 'kernel_regularizer':'l1_l2' if regularize_bool else None}

    deconv_params = lambda n, pad, regularize_bool: {'filters':n*C1, 'use_bias':True, 'padding':'same', 'data_format': data_format, 'output_padding':(pad,pad) if pad != 0 else None, 'kernel_regularizer':'l1_l2' if regularize_bool else None}

    bn_params = {'momentum':0.9, 'epsilon':1e-5, 'axis':axis}
    drop_params = lambda d: {'rate':d}

    feature=64
    #----------------------------------------------------------------------- Herschel -> SR   -----------------------------------------------------------------------
    inp = Input(shape=shape)  
    x = Conv2D(filters=feature*4, kernel_size=4, strides=(2,2), data_format="channels_first", 
                            padding='SAME', kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(inp)
    skip1 = x
    x = BatchNormalization(**bn_params)(x) 
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(filters=feature*8, kernel_size=4, strides=(2,2), data_format="channels_first", 
                            padding='SAME', kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x)
    skip2 = x
    x = BatchNormalization(**bn_params)(x) 
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(filters=feature*8, kernel_size=4, strides=(2,2), data_format="channels_first", 
                            padding='SAME', kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x)
    skip3 = x
    x = BatchNormalization(**bn_params)(x) 
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(filters=feature*8, kernel_size=4, strides=(2,2), data_format="channels_first", 
                            padding='SAME', kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x)
    skip4 = x
    x = BatchNormalization(**bn_params)(x) 
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(filters=feature*8, kernel_size=4, strides=(2,2), data_format="channels_first", 
                            padding='SAME', kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x)
    skip5 = x
    x = BatchNormalization(**bn_params)(x) 
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(filters=feature*8, kernel_size=4, strides=(2,2), data_format="channels_first", 
                            padding='SAME', kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x)
    x = BatchNormalization(**bn_params)(x) 
    x = LeakyReLU()(x)
    
    
    # 4x4
    x = Conv2DTranspose(filters=feature*8, kernel_size=4, strides=(2,2), data_format="channels_first", padding='same',                                      
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x)
    x = BatchNormalization(**bn_params)(x)
    x = Dropout(0.5)(x)
    
    x = tf.concat([x, skip5], axis)
    x = LeakyReLU()(x)

    # 7x7
    x = Conv2DTranspose(filters=feature*8, kernel_size=4, strides=(2,2), data_format="channels_first",                                      
                              output_padding=(1,1), padding='same',                               
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x)
    x = BatchNormalization(**bn_params)(x) 
    x = Dropout(0.5)(x)
    
    x = tf.concat([x, skip4], axis)
    x = LeakyReLU()(x)
    
    # 14x14
    x = Conv2DTranspose(filters=feature*8, kernel_size=4, strides=(2,2), data_format="channels_first", padding='same',                                      
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x)
    x = BatchNormalization(**bn_params)(x) 
    x = Dropout(0.5)(x)
    
    x = tf.concat([x, skip3], axis)
    x = LeakyReLU()(x)
    
    # 27x27
    x = Conv2DTranspose(filters=feature*8, kernel_size=4, strides=(2,2), data_format="channels_first",                                      
                               output_padding=(1,1), padding='same',                                
                               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x)
    x = BatchNormalization(**bn_params)(x) 
    #build_15 = layers.Dropout(0.5)(build_14)
    
    x = tf.concat([x, skip2], axis)
    x = LeakyReLU()(x)
    
    # 53x53
    x = Conv2DTranspose(filters=feature*4, kernel_size=4, strides=(2,2), data_format="channels_first",                                      
                               output_padding=(1,1), padding='same',                                
                               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x)
    x = BatchNormalization(**bn_params)(x) 
    #build_19 = layers.Dropout(0.5)(build_18)
    
    x = tf.concat([x, skip1], axis)
    x = LeakyReLU()(x)
    
    # 106x106
    x = Conv2DTranspose(filters=feature*2, kernel_size=4, strides=(2,2), data_format="channels_first", padding='same',                                      
                               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x)
    x = BatchNormalization(**bn_params)(x) 
    #build_23 = layers.Dropout(0.5)(build_22)
    x = LeakyReLU()(x)
    
    # 212x212
    x = Conv2DTranspose(filters=feature, kernel_size=4, strides=(2,2), data_format="channels_first", padding='same',                                      
                               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x)
    x = BatchNormalization(**bn_params)(x) 
    #build_27 = layers.Dropout(0.5)(build_26)
    
    x = LeakyReLU()(x)

    # 424x424        
    x = Conv2DTranspose(filters=1, kernel_size=4, strides=(2,2), data_format="channels_first", padding='same',                                
                               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x) #tf.nn.tanh(build_22)
    
    x = tf.keras.activations.sigmoid(x) #keras.layers.LeakyReLU(alpha=0.3)(build_34) #activations.leakyrelu(build_34)
    
    build_model = tf.keras.Model(inp, x)
    return build_model

def Discriminator(input_shape, conditioning_shape, data_format, C1):
    axis = 1 if input_shape[0] != input_shape[1] else 3

    #const = ClipConstraint(clip)
    const = None
    # Block parameters
    conv_params = lambda n, s, K: {'filters':n*C1, 'kernel_size': (K, K), 'strides':(s,s), 'use_bias':True, 'padding':'same', 'data_format': data_format, 'kernel_constraint':const}
    dense_params = lambda n: {'units':n, 'use_bias':True}
    bn_params = {'momentum':0.9, 'epsilon':1e-5, 'axis':axis}
    act_params = {'alpha':0.2}
    drop_params = lambda d: {'rate':d}

    # Model Blocks
    inp = Input(shape=input_shape)
    x = ConvolutionBlock(conv_params(2, 1, 4), bn_params, act_params, drop_params(0), inp, use_bn=False, use_ln=True)
    x = ConvolutionBlock(conv_params(2, 2, 4), bn_params, act_params, drop_params(0), x, use_bn=False, use_ln=True)
    x = ConvolutionBlock(conv_params(4, 1, 4), bn_params, act_params, drop_params(0), x, use_bn=False, use_ln=True)
    x = ConvolutionBlock(conv_params(4, 2, 4), bn_params, act_params, drop_params(0), x, use_bn=False, use_ln=True)
    
    # Input the conditional data
    inp_conditional = Input(shape=conditioning_shape)
    x = tf.concat([x, inp_conditional], axis)
    
    x = ConvolutionBlock(conv_params(8, 1, 4), bn_params, act_params, drop_params(0), x, use_bn=False, use_ln=True)
    x = ConvolutionBlock(conv_params(8, 2, 4), bn_params, act_params, drop_params(0), x, use_bn=False, use_ln=True)
    x = ConvolutionBlock(conv_params(16, 1, 4), bn_params, act_params, drop_params(0.), x, use_bn=False, use_ln=True)
    x = ConvolutionBlock(conv_params(16, 2, 4), bn_params, act_params, drop_params(0.2), x, use_bn=False, use_ln=True)
    x = ConvolutionBlock(conv_params(16, 2, 4), bn_params, act_params, drop_params(0.2), x, use_bn=False, use_ln=True)
    x = ConvolutionBlock(conv_params(16, 2, 4), bn_params, act_params, drop_params(0.2), x, use_bn=False, use_ln=True)
    x = Flatten()(x)
    x = DenseBlock(dense_params(128), bn_params, act_params, drop_params(0.), x, use_bn=False)

    #D1 = DenseBlock(dense_params(2), bn_params, act_params, drop_params(0), FCB4, use_bn=False)
    x = Dense(1, activation='linear')(x)
    return tf.keras.Model(inputs=[inp, inp_conditional], outputs=x)