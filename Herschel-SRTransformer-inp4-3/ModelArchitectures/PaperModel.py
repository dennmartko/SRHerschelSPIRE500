import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Input, Dense, LeakyReLU, Flatten, Conv2DTranspose
from ModelArchitectures.ModelBlocks import ConvolutionBlock, ConvMultiScaleBlock, DeConvMultiScaleBlock, CAM, BottleneckBlock


def Generator(shape, data_format, C1, K, multipliers):
    axis = 1 if shape[0] != shape[1] else 3
    # Block parameters
    shape = shape
    conv_params = lambda n, regularize_bool: {'filters':n*C1, 'kernel_initializer':tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None), 'bias_initializer':'zeros', 'use_bias':True, 'padding':'same', 'data_format': data_format, 'kernel_regularizer':'l1_l2' if regularize_bool else None}
    conv_params_final = lambda regularize_bool: {'filters':1, 'kernel_initializer':tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None), 'bias_initializer':'zeros', 'use_bias':True, 'padding':'same', 'data_format': data_format, 'kernel_regularizer':'l1_l2' if regularize_bool else None}

    deconv_params = lambda n, pad, regularize_bool: {'filters':n*C1, 'kernel_initializer':tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None), 'bias_initializer':'zeros', 'use_bias':True, 'padding':'same', 'data_format': data_format, 'output_padding':(pad,pad) if pad != 0 else None, 'kernel_regularizer':'l1_l2' if regularize_bool else None}

    bn_params = {'momentum':0.9, 'epsilon':1e-5, 'axis':axis}
    drop_params = lambda d: {'rate':d}

    # Model Blocks
    inp = Input(shape=shape)

    # Down Sample
    x, skip1 = ConvMultiScaleBlock(conv_params(multipliers[1], False), conv_params(multipliers[1], False), bn_params, drop_params(0.), inp, axis, 0) # 53, 128
    x, skip2 = ConvMultiScaleBlock(conv_params(multipliers[1], False), conv_params(multipliers[2], False), bn_params, drop_params(0.), x, axis, 1) # 27, 256
    x, skip3 = ConvMultiScaleBlock(conv_params(multipliers[2], False), conv_params(multipliers[3], False), bn_params, drop_params(0.), x, axis, 2) # 14, 512
    x, skip4 = ConvMultiScaleBlock(conv_params(multipliers[3], False), conv_params(multipliers[3], False), bn_params, drop_params(0.3), x, axis, 3) # 7, 512
    x, skip5 = ConvMultiScaleBlock(conv_params(multipliers[3], False), conv_params(multipliers[3], False), bn_params, drop_params(0.3), x, axis, 4) # 4, 512
    x, skip6 = ConvMultiScaleBlock(conv_params(multipliers[3], False), conv_params(multipliers[3], False), bn_params, drop_params(0.5), x, axis, 5) # 2, 512

    # Up Sample
    X = BottleneckBlock(conv_params(multipliers[3], False), deconv_params(multipliers[3], 0, False), bn_params, drop_params(0.5), x, axis)
    x = tf.concat([X, skip6], axis=axis)
    X = DeConvMultiScaleBlock(conv_params(multipliers[3], False), deconv_params(multipliers[3], 1, False), bn_params, drop_params(0.3), x, axis, residual = X, extra_skip = None) #7
    x = tf.concat([X, skip5], axis=axis)
    X = DeConvMultiScaleBlock(conv_params(multipliers[3], False), deconv_params(multipliers[3], 0, False), bn_params, drop_params(0.3), x, axis, residual = X, extra_skip = None) #14
    x = tf.concat([X, skip4], axis=axis)
    X = DeConvMultiScaleBlock(conv_params(multipliers[3], False), deconv_params(multipliers[2], 1, False), bn_params, drop_params(0.), x, axis, residual = X, extra_skip = None) # 27
    x = tf.concat([X, skip3], axis=axis)
    X = DeConvMultiScaleBlock(conv_params(multipliers[2], False), deconv_params(multipliers[1], 1, False), bn_params, drop_params(0.), x, axis, residual = X, extra_skip = None) # 53
    x = tf.concat([X, skip2], axis=axis)
    X = DeConvMultiScaleBlock(conv_params(multipliers[1], False), deconv_params(multipliers[1], 0, False), bn_params, drop_params(0.), x, axis, residual = X, extra_skip = None) # 106
    x = tf.concat([X, skip1], axis=axis)
    X = DeConvMultiScaleBlock(conv_params(multipliers[1], False), deconv_params(multipliers[1], 0, False), bn_params, drop_params(0.), x, axis, residual = X, extra_skip = inp) # 212
    X = DeConvMultiScaleBlock(conv_params(multipliers[1], False), deconv_params(multipliers[0], 0, False), bn_params, drop_params(0.), X, axis, residual = X, extra_skip = None) # 424

    # Out track
    x = Conv2D(**conv_params(multipliers[0], False), kernel_size=(3, 3), strides=(1,1))(X)
    x = BatchNormalization(**bn_params)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    # x = CAM(conv_params(multipliers[0], False), x, axis, num_heads=4)
    out = Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), use_bias=True, padding='same', data_format=data_format)(x)
    out = tf.keras.activations.sigmoid(out)
    return tf.keras.Model(inp, out)