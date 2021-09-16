from classification_models.tfkeras import Classifiers
import tensorflow as tf
from tensorflow import keras
import numpy as np

def GetRegressionModel(image_width, image_height, n_labels):
    L2_REG = 1.25e-5
    shape = (image_height,image_width,3)

    # Charge le ResNet18
    ResNet18, preprocess_input = Classifiers.get("resnet18")
    base_model = ResNet18(input_shape=shape, weights="imagenet", include_top=False)
    
    
    # Ajout de la régularisation L2
    for layer in base_model.layers:
        layer.kernel_regularizer = tf.keras.regularizers.l2()
    out = tf.keras.models.model_from_json(base_model.to_json())
    out.set_weights(base_model.get_weights())

    # Ajout des déconvolutions
    c5 = tf.keras.layers.Dropout(rate=0.5)(out.get_layer("relu1").output)
    dcn = tf.keras.layers.Conv2DTranspose(256, kernel_size=(4,4), strides=(2,2), padding="same",
                                          use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
                                          kernel_initializer="he_uniform", name="deconv1")(c5)

    dcn = tf.keras.layers.BatchNormalization()(dcn)
    dcn = tf.keras.layers.Activation("relu")(dcn)
    dcn = tf.keras.layers.Conv2DTranspose(128, kernel_size=(4,4), strides=(2,2), padding="same",
                                          use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
                                          kernel_initializer="he_uniform", name="deconv2")(dcn)

    dcn = tf.keras.layers.BatchNormalization()(dcn)
    dcn = tf.keras.layers.Activation("relu")(dcn)
    dcn = tf.keras.layers.Conv2DTranspose(64, kernel_size=(4,4), strides=(2,2), padding="same",
                                          use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
                                          kernel_initializer="he_uniform", name="features")(dcn)
    dcn = tf.keras.layers.BatchNormalization()(dcn)
    features = tf.keras.layers.Activation("relu")(dcn)
    
    # Création de la heatmap
    output_heatmap = tf.keras.layers.Conv2D(
        64,(3, 3),
        padding="same",
        use_bias=False,
        kernel_initializer=tf.keras.initializers.RandomNormal(0.01),
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        name="heatmap_conv2D",
    )(features)
    output_heatmap = tf.keras.layers.BatchNormalization(name="heatmap_norm")(output_heatmap)
    output_heatmap = tf.keras.layers.Activation("relu", name="heatmap_activ")(output_heatmap)
    output_heatmap = tf.keras.layers.Conv2D(
        n_labels,(1, 1),
        padding="valid",
        activation=tf.nn.sigmoid,
        kernel_initializer=tf.keras.initializers.RandomNormal(0.01),
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        bias_initializer=tf.constant_initializer(-np.log((1.0 - 0.1) / 0.1)),
        name="heatmap",
    )(output_heatmap)
    return tf.keras.models.Model(inputs=out.input, outputs=output_heatmap)

