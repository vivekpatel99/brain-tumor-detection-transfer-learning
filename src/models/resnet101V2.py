import math

import keras
import tensorflow as tf
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    LeakyReLU,
    Reshape,
)
from keras.regularizers import l2
from pkg_resources import add_activation_listener


def feature_extractor(inputs)-> keras.Model:
    resnet101 = tf.keras.applications.ResNet101V2(
        include_top = False, 
        weights = "imagenet",    
        input_tensor=inputs
    )
    resnet101.trainable = True
    feature_eloc_branchtractor = resnet101.output

    return feature_eloc_branchtractor

def bounding_box_regression(feature_extr, num_classes:int)->keras.Layer:
    bbox_shape=4

    # Pool and process
    loc_branch = GlobalAveragePooling2D()(feature_extr)
    loc_branch = Dense(1024, activation='leaky_relu', name='bounding_box_dense_1024')(loc_branch)
    loc_branch = Dense(512, activation='leaky_relu', name='bounding_box_dense_512')(loc_branch)
    loc_branch = Dropout(0.2)(loc_branch)
    loc_branch = Dense(256, activation='leaky_relu',name='bounding_box_dense_256')(loc_branch)
    loc_branch = Dense(128,  name='bounding_box_dense_128')(loc_branch)
    loc_branch = Dropout(0.2)(loc_branch)

    # Add sigmoid activation to ensure output is between 0 and 1
    bboloc_branch_reg_output = Dense(units=bbox_shape*num_classes, 
                                     activation='sigmoid', 
                                     name='bounding_box_reg_output')(loc_branch)
    return Reshape((num_classes, 4), name='bounding_box')(bboloc_branch_reg_output)

def classifer(feature_extr, num_classes, l2_reg=0.01)->keras.Model:
    # cls_branch = Conv2D(256, (1, 1), activation='relu', name='classification_conv2d_256')(feature_eloc_branchtr)
    cls_branch = GlobalAveragePooling2D()(feature_extr)
    cls_branch = Dense(256, activation='relu', name='classification_dense_256')(cls_branch)
    cls_branch = Dropout(0.5)(cls_branch)
    return Dense(units=num_classes, activation='sigmoid', 
                                 kernel_regularizer=l2(l2_reg), 
                                 name = 'classification')(cls_branch)

def final_model(input_shape:tuple, num_classes:int)-> keras.Model:
    
    inputs = Input(shape=input_shape)

    _feature_extractor_output = feature_extractor(inputs)
   
    # classification branch
    classification_output = classifer(_feature_extractor_output, num_classes)

    # regression branch (preserves spatial info)
    bbox_reg_output = bounding_box_regression(_feature_extractor_output, num_classes)

    
    return keras.Model(inputs=inputs, 
                          outputs=[classification_output, 
                                   bbox_reg_output])


def resnet101_classifier(input_shape:tuple, num_classes:int)-> keras.Model:
    inputs = Input(shape=input_shape)

    _feature_extractor_output = feature_extractor(inputs)
    loc_branch = GlobalAveragePooling2D()(_feature_extractor_output)
    loc_branch = Dense(1024, activation='relu')(loc_branch)
    loc_branch = Dropout(0.5)(loc_branch)
    classification_output = classifer(loc_branch, num_classes)

    return keras.Model(inputs=inputs, 
                          outputs=classification_output)



def resnet101_regressor(input_shape:tuple, num_classes:int)-> keras.Model:
    inputs = Input(shape=input_shape)

    _feature_extractor_output = feature_extractor(inputs)

    bboloc_branch_reg_output = bounding_box_regression(_feature_extractor_output, num_classes)

    return keras.Model(inputs=inputs, 
                          outputs=bboloc_branch_reg_output)