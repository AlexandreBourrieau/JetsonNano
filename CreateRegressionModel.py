from classification_models.tfkeras import Classifiers
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Modèle basé sur CenterNet
# Sans pré-convolution lors du sur-échantillonnage 
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


# Modèle basé sur CenterNet
# Avec pré-convolution lors du sur-échantillonnage 
def GetRegressionModel2(image_width, image_height, n_labels):
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
    up = tf.keras.layers.Conv2DTranspose(256, kernel_size=(4,4), strides=(2,2), padding="same",
                                          use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
                                          kernel_initializer="he_uniform", name="deconv1")(c5)

    up = tf.keras.layers.BatchNormalization()(up)
    up = tf.keras.layers.Activation("relu")(up)

    up = tf.keras.layers.Conv2D(256,(3, 3),padding="same",kernel_regularizer=tf.keras.regularizers.l2(L2_REG),name="conv33_2",)(up)
    up = tf.keras.layers.Conv2DTranspose(128, kernel_size=(4,4), strides=(2,2), padding="same",
                                          use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
                                          kernel_initializer="he_uniform", name="deconv2")(up)

    up = tf.keras.layers.BatchNormalization()(up)
    up = tf.keras.layers.Activation("relu")(up)

    up = tf.keras.layers.Conv2D(128,(3, 3),padding="same",kernel_regularizer=tf.keras.regularizers.l2(L2_REG),name="conv33_3",)(up)
    up = tf.keras.layers.Conv2DTranspose(64, kernel_size=(4,4), strides=(2,2), padding="same",
                                          use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
                                          kernel_initializer="he_uniform", name="features")(up)
    up = tf.keras.layers.BatchNormalization()(up)
    features = tf.keras.layers.Activation("relu")(up)
    
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

# Modèle basé sur TTFNet
def GetRegressionModel_TTFNet(image_width, image_height, n_labels):
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
    
    c2 = tf.keras.layers.Dropout(rate=0.2)(out.get_layer("stage2_unit1_relu1").output)
    c3 = tf.keras.layers.Dropout(rate=0.4)(out.get_layer("stage3_unit1_relu1").output)
    c4 = tf.keras.layers.Dropout(rate=0.4)(out.get_layer("stage4_unit1_relu1").output)
    c5 = tf.keras.layers.Dropout(rate=0.5)(out.get_layer("relu1").output)

    
    p3_out = keras.layers.Conv2D(256, 3, 3, "same")(c3)
    p4_out = keras.layers.Conv2D(256, 3, 3, "same")(c4)
    p5_out = keras.layers.Conv2D(256, 3, 3, "same")(c5)
    

   
        
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
