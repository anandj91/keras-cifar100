from tensorflow.python.keras import layers
from tensorflow.python.keras import Model
from tensorflow.python.keras.regularizers import l2

import sys
sys.path.append("..")
from params import *


def identity_block(input_tensor, kernel_size, filters, stage, block, weight_decay=5e-4, use_bias=False):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=use_bias)(input_tensor)
    x = layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=use_bias)(x)
    x = layers.BatchNormalization(name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=use_bias)(x)
    x = layers.BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               weight_decay=5e-4,
               use_bias=False):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=use_bias)(input_tensor)
    x = layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=use_bias)(x)
    x = layers.BatchNormalization(name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=use_bias)(x)
    x = layers.BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1',
                             kernel_regularizer=l2(weight_decay),
                             use_bias=use_bias)(input_tensor)
    shortcut = layers.BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def resnet(num_block,
           input_shape=(img_size, img_size, channel),
           classes=100,
           weight_decay=5e-4,
           use_bias=False):
    main_input = layers.Input(input_shape, name='resnet_input')
   
    x = layers.Conv2D(64, (3, 3),
                      padding='same',
                      kernel_initializer='he_normal',
                      name='conv1',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=use_bias)(main_input)
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.Activation('relu')(x)

    ascii_a = 97
    
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    for i in range(num_block[0] - 1):
        block_id = chr(ascii_a + i + 1)
        x = identity_block(x, 3, [64, 64, 256], stage=2, block=block_id,
                           weight_decay=weight_decay, use_bias=use_bias)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a',
                   weight_decay=weight_decay, use_bias=use_bias)
    for i in range(num_block[1] - 1):
        block_id = chr(ascii_a + i + 1)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block=block_id,
                           weight_decay=weight_decay, use_bias=use_bias)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a',
                   weight_decay=weight_decay, use_bias=use_bias)
    for i in range(num_block[2] - 1):
        block_id = chr(ascii_a + i + 1)
        if num_block[2] > 10:
            block_id = 'b' + str(i + 1)
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=block_id,
                           weight_decay=weight_decay, use_bias=use_bias)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a',
                   weight_decay=weight_decay, use_bias=use_bias)
    for i in range(num_block[0] - 1):
        block_id = chr(ascii_a + i + 1)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block=block_id,
                           weight_decay=weight_decay, use_bias=use_bias)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(classes, activation='softmax', name='fc')(x) 
    
    model = Model(main_input, x)

    return model


def resnet50():
    return resnet([3, 4, 6, 3])


def resnet101():
    return resnet([3, 4, 23, 3])


def resnet152():
    return resnet([3, 8, 36, 3])


if __name__ == "__main__":
    model = resnet152()
    model.summary()
