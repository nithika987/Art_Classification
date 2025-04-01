import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Attention, Reshape
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.layers import Resizing

def build_model(img_height=300, img_width=300, num_classes=8):
    inputs = Input(shape=(img_height, img_width, 3))
    
    # Data Augmentation
    data_augmentation_layers = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.5),
        layers.RandomZoom(0.5),
        layers.RandomBrightness(0.5),
        layers.RandomContrast(0.5),
        layers.RandomTranslation(0.2, 0.2),
        layers.RandomCrop(300, 300),
        layers.GaussianNoise(0.1)
    ])
    x = data_augmentation_layers(inputs)
    x = applications.efficientnet.preprocess_input(x)
    
    # Feature Extraction via EfficientNet
    efficient_net = EfficientNetV2S(include_top=False, weights='imagenet', input_shape=(300, 300, 3))
    efficient_net.trainable = True  
    for layer in efficient_net.layers[:-int(len(efficient_net.layers) * 0.5)]:
        layer.trainable = False
    
    x = efficient_net(x)
    x = layers.Conv2D(512, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    
    # Reshape for ConvLSTM
    x = Reshape((1, 10, 10, 512))(x)  # Single time step
    x = ConvLSTM2D(256, (3, 3), padding="same", return_sequences=True, activation="relu")(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(128, (3, 3), padding="same", return_sequences=False, activation="relu")(x)
    
    # Flatten spatial dimensions correctly
    x_shape = x.shape
    x_flattened = Reshape((x_shape[1] * x_shape[2], x_shape[3]))(x)
    
    # Apply attention
    attention = Attention()([x_flattened, x_flattened])
    attention = Reshape((x_shape[1], x_shape[2], x_shape[3]))(attention)
    x = layers.Add()([x, attention])
    
    # Feature Pyramid
    p3 = layers.Conv2D(128, (1, 1), activation="relu")(x)
    p3 = Resizing(20, 20)(p3)
    
    p4 = layers.Conv2D(128, (1, 1), activation="relu")(x)
    p4 = Resizing(20, 20)(p4)
    
    x = layers.Concatenate()([p3, p4])
    x = layers.GlobalAveragePooling2D()(x)
    
    # Regularization
    x = layers.Dropout(0.4)(x)
    
    # Final Classification Layer
    outputs = layers.Dense(num_classes, activation='softmax', name='classification_layer')(x)
    
    # Define Model
    model = Model(inputs, outputs)
    return model
