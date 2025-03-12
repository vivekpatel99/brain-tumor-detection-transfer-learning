import keras
import tensorflow as tf


### Define ResNet50 as a Feature Extractor
def feature_extractor(inputs)-> tf.keras.Model:
    resnet50 = tf.keras.applications.ResNet50(
        include_top = False, 
        weights = "imagenet",    
        input_tensor=inputs
    )
    resnet50.trainable = True
    for layer in resnet50.layers[:140]: #example number of layers to freeze
        layer.trainable = False
    feature_extractor = resnet50.output
    return feature_extractor


### Define Dense Layers
def dense_layers(features)->tf.keras.Layer:
    x = keras.layers.Conv2D(filters=256, kernel_size=(1, 1), activation='relu')(features) # 1x1 conv
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=1024, activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(units=512, activation='relu', kernel_regularizer='l2')(x)
    return x

### Define Bounding Box Regression
def bounding_box_regression(x, num_classes:int)->tf.keras.Layer:
    bbox_shape=4
    bounding_box_regression_output = tf.keras.layers.Dense(units=bbox_shape*num_classes, name='_bounding_box', activation='linear')(x)
    reshape_bbox = tf.keras.layers.Reshape(
        (num_classes, 4),  # Not hard-coded
        name='bounding_box'
    )(bounding_box_regression_output)
    return reshape_bbox

###Define Classifier Layer
def classifer(inputs, num_classes)->tf.keras.Model:
    return tf.keras.layers.Dense(units=num_classes, activation='sigmoid', name = 'classification')(inputs)

def final_model(input_shape:tuple, num_classes:int)->tf.keras.Model:
    
    inputs = tf.keras.layers.Input(shape=input_shape)

    _feature_extractor = feature_extractor(inputs)
   
    dense_output = dense_layers(_feature_extractor)

    bounding_box_regression_output = bounding_box_regression(dense_output, num_classes)

    classification_output = classifer(dense_output, num_classes)

    return tf.keras.Model(inputs=inputs, 
                          outputs=[classification_output, 
                                   bounding_box_regression_output])