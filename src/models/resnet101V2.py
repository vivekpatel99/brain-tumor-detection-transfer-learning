import math

import keras
import tensorflow as tf


def feature_extractor(inputs)-> keras.Model:
    resnet101 = tf.keras.applications.ResNet101V2(
        include_top = False, 
        weights = "imagenet",    
        input_tensor=inputs
    )
    resnet101.trainable = True

    total_layers = len(resnet101.layers)
    trainable_layers = math.ceil(total_layers * 0.20) 
    # Then freeze all layers except the last layers
    for layer in resnet101.layers[:-trainable_layers]:
        layer.trainable = False

    feature_extractor = resnet101.output

    return feature_extractor

def bounding_box_regression(feature_extr, num_classes:int)->keras.Layer:
    bbox_shape=4
    loc_branch = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='bounding_box_conv2d_256')(feature_extr)
    loc_branch = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='bounding_box_conv2d_256_2')(loc_branch)
    loc_branch = keras.layers.Flatten()(loc_branch)
    loc_branch = keras.layers.Dense(1024, activation='relu')(loc_branch)

    # Add sigmoid activation to ensure output is between 0 and 1
    bbox_reg_output = tf.keras.layers.Dense(units=bbox_shape*num_classes, name='bounding_box_reg_output')(loc_branch)
    return tf.keras.layers.Reshape((num_classes, 4), name='bounding_box')(bbox_reg_output)

def classifer(feature_extr, num_classes, l2_reg=0.01)->keras.Model:
    cls_branch = keras.layers.Conv2D(256, (1, 1), activation='relu', name='classification_conv2d_256')(feature_extr)
    cls_branch = keras.layers.GlobalAveragePooling2D()(cls_branch)
    cls_branch = keras.layers.Dense(1024, activation='relu', name='classification_dense_1024')(cls_branch)

    return tf.keras.layers.Dense(units=num_classes, activation='sigmoid', 
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg), 
                                 name = 'classification')(cls_branch)

def final_model(input_shape:tuple, num_classes:int)-> keras.Model:
    
    inputs = tf.keras.layers.Input(shape=input_shape)

    _feature_extractor = feature_extractor(inputs)
   
    # classification branch
    classification_output = classifer(_feature_extractor, num_classes)

    # regression branch (preserves spatial info)
    bbox_reg_output = bounding_box_regression(_feature_extractor, num_classes)

    
    return keras.Model(inputs=inputs, 
                          outputs=[classification_output, 
                                   bbox_reg_output])


def resnet101_classifier(input_shape:tuple, num_classes:int)-> keras.Model:
    inputs = keras.layers.Input(shape=input_shape)

    _feature_extractor = feature_extractor(inputs)
    x = keras.layers.GlobalAveragePooling2D()(_feature_extractor)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    classification_output = classifer(x, num_classes)

    return keras.Model(inputs=inputs, 
                          outputs=classification_output)



def resnet101_regressor(input_shape:tuple, num_classes:int)-> keras.Model:
    inputs = keras.layers.Input(shape=input_shape)

    _feature_extractor = feature_extractor(inputs)

    bbox_reg_output = bounding_box_regression(_feature_extractor, num_classes)

    return keras.Model(inputs=inputs, 
                          outputs=bbox_reg_output)