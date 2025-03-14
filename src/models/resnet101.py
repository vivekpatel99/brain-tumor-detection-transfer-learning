import keras
import tensorflow as tf


### Define ResNet50 as a Feature Extractor
def feature_extractor(inputs)-> keras.Model:
    resnet101 = tf.keras.applications.ResNet101(
        include_top = False, 
        weights = "imagenet",    
        input_tensor=inputs
    )
    resnet101.trainable = True
    # Determine the number of layers to unfreeze dynamically
    total_layers = len(resnet101.layers)
    unfreeze_percentage = 0.5  # unfreezing 50% of the layers
    layers_to_unfreeze = int(total_layers * unfreeze_percentage)
    
    # Unfreeze the last 'layers_to_unfreeze' layers
    for layer in resnet101.layers[:total_layers - layers_to_unfreeze]:
        layer.trainable = False

    print(f"Total layers in ResNet101: {total_layers}")
    print(f"Unfreezing the last {layers_to_unfreeze} layers ({unfreeze_percentage*100:.0f}% of total layers)")
    
    feature_extractor = resnet101.output
    return feature_extractor


### Define Dense Layers
def dense_layers(features)-> keras.Layer:
    x = keras.layers.Conv2D(filters=256, kernel_size=(1, 1), activation='relu')(features) # 1x1 conv
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(units=1024, activation='relu', kernel_regularizer='l2')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(units=512, activation='relu', kernel_regularizer='l2')(x)
    return x

### Define Bounding Box Regression
def bounding_box_regression(x, num_classes:int)->keras.Layer:
    bbox_shape=4
    bbox_reg_output = tf.keras.layers.Dense(units=bbox_shape*num_classes, name='_bounding_box', activation='linear')(x)
    reshape_bbox = tf.keras.layers.Reshape(
        (num_classes, 4),  # Not hard-coded
        name='bounding_box'
    )(bbox_reg_output)
    return reshape_bbox

###Define Classifier Layer
def classifer(inputs, num_classes, l2_reg=0.01)->keras.Model:
    return tf.keras.layers.Dense(units=num_classes, activation='sigmoid', 
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg), 
                                 name = 'classification')(inputs)

def final_model(input_shape:tuple, num_classes:int)-> keras.Model:
    
    inputs = tf.keras.layers.Input(shape=input_shape)

    _feature_extractor = feature_extractor(inputs)
   
    dense_output = dense_layers(_feature_extractor)

    bbox_reg_output = bounding_box_regression(dense_output, num_classes)

    classification_output = classifer(dense_output, num_classes)

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
    x = keras.layers.GlobalAveragePooling2D()(_feature_extractor)
    bbox_reg_output = bounding_box_regression(x, num_classes)

    return keras.Model(inputs=inputs, 
                          outputs=bbox_reg_output)