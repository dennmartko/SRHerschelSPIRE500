######################
###     IMPORTS    ###
######################
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Input, Dense, LeakyReLU, Flatten, Conv2DTranspose, Dropout

def Generator(shape, data_format):
    axis = 1 if shape[0] != shape[1] else 3
    # Block parameters
    shape = shape

    feature=64
    #----------------------------------------------------------------------- Herschel -> SR   -----------------------------------------------------------------------
    inp = Input(shape=shape)  
    x = Conv2D(filters=feature*4, kernel_size=4, strides=(2,2), data_format=data_format, 
                            padding='SAME', kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(inp)
    skip1 = x
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=axis)(x) 
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(filters=feature*8, kernel_size=4, strides=(2,2), data_format=data_format, 
                            padding='SAME', kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x)
    skip2 = x
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=axis)(x) 
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(filters=feature*8, kernel_size=4, strides=(2,2), data_format=data_format, 
                            padding='SAME', kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x)
    skip3 = x
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=axis)(x) 
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(filters=feature*8, kernel_size=4, strides=(2,2), data_format=data_format, 
                            padding='SAME', kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x)
    skip4 = x
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=axis)(x) 
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(filters=feature*8, kernel_size=4, strides=(2,2), data_format=data_format, 
                            padding='SAME', kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x)
    skip5 = x
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=axis)(x) 
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(filters=feature*8, kernel_size=4, strides=(2,2), data_format=data_format, 
                            padding='SAME', kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=axis)(x) 
    x = LeakyReLU()(x)
    
    
    # 4x4
    x = Conv2DTranspose(filters=feature*8, kernel_size=4, strides=(2,2), data_format=data_format, padding='same',                                      
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=axis)(x)
    x = Dropout(0.5)(x)
    
    x = tf.concat([x, skip5], axis)
    x = LeakyReLU()(x)

    # 7x7
    x = Conv2DTranspose(filters=feature*8, kernel_size=4, strides=(2,2), data_format=data_format,                                      
                              output_padding=(1,1), padding='same',                               
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=axis)(x) 
    x = Dropout(0.5)(x)
    
    x = tf.concat([x, skip4], axis)
    x = LeakyReLU()(x)
    
    # 14x14
    x = Conv2DTranspose(filters=feature*8, kernel_size=4, strides=(2,2), data_format=data_format, padding='same',                                      
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=axis)(x) 
    x = Dropout(0.5)(x)
    
    x = tf.concat([x, skip3], axis)
    x = LeakyReLU()(x)
    
    # 27x27
    x = Conv2DTranspose(filters=feature*8, kernel_size=4, strides=(2,2), data_format=data_format,                                      
                               output_padding=(1,1), padding='same',                                
                               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=axis)(x) 
    #build_15 = layers.Dropout(0.5)(build_14)
    
    x = tf.concat([x, skip2], axis)
    x = LeakyReLU()(x)
    
    # 53x53
    x = Conv2DTranspose(filters=feature*4, kernel_size=4, strides=(2,2), data_format=data_format,                                      
                               output_padding=(1,1), padding='same',                                
                               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=axis)(x) 
    #build_19 = layers.Dropout(0.5)(build_18)
    
    x = tf.concat([x, skip1], axis)
    x = LeakyReLU()(x)
    
    # 106x106
    x = Conv2DTranspose(filters=feature*2, kernel_size=4, strides=(2,2), data_format=data_format, padding='same',                                      
                               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=axis)(x) 
    #build_23 = layers.Dropout(0.5)(build_22)
    x = LeakyReLU()(x)
    
    # 212x212
    x = Conv2DTranspose(filters=feature, kernel_size=4, strides=(2,2), data_format=data_format, padding='same',                                      
                               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=axis)(x)
    #build_27 = layers.Dropout(0.5)(build_26)
    
    x = LeakyReLU()(x)

    # 424x424        
    x = Conv2DTranspose(filters=1, kernel_size=4, strides=(2,2), data_format=data_format, padding='same',                                
                               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer='zeros')(x) #tf.nn.tanh(build_22)
    
    x = tf.keras.activations.sigmoid(x) #keras.layers.LeakyReLU(alpha=0.3)(build_34) #activations.leakyrelu(build_34)
    
    build_model = tf.keras.Model(inp, x)
    return build_model