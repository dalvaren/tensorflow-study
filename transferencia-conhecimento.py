import os
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %matplotlib inline
print(tf.__version__)

# https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip

dataset_path = "./cats_and_dogs_filtered.zip"
zip_object = zipfile.ZipFile(file=dataset_path, mode="r")
zip_object.extractall("./")
zip_object.close()

dataset_path_new = "./cats_and_dogs_filtered"
train_dir = os.path.join(dataset_path_new, "train")
validation_dir = os.path.join(dataset_path_new, "validation")

# construindo o modelo
img_shape = (128, 128, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape = img_shape, 
                                               include_top = False,
                                               weights = "imagenet")
base_model.summary()

# congelando modelo base
base_model.trainable = False

# Definindo o cabeçalho personalizado da rede neural
base_model.output
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
# global_average_layer
prediction_layer = tf.keras.layers.Dense(units = 1, activation = "sigmoid")(global_average_layer)

# Definindo o modelo
model = tf.keras.models.Model(inputs = base_model.input, outputs = prediction_layer)
model.summary()

# compilando o modelo
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr = 0.0001),
              loss="binary_crossentropy", metrics = ["accuracy"])

# redimensionando imagens
data_gen_train = ImageDataGenerator(rescale=1/255.)
data_gen_valid = ImageDataGenerator(rescale=1/255.)
train_generator = data_gen_train.flow_from_directory(train_dir, target_size=(128,128), batch_size=128, class_mode="binary")
valid_generator = data_gen_train.flow_from_directory(validation_dir, target_size=(128,128), batch_size=128, class_mode="binary")

# treinando o modelo
model.fit_generator(train_generator, epochs=5, validation_data=valid_generator)

# Avaliação do modelo de transferência de aprendizagem
valid_loss, valid_accuracy = model.evaluate_generator(valid_generator)
print(valid_accuracy)

# Descongelando algumas camadas do topo do modelo base
base_model.trainable = True
fine_tuning_at = 100
for layer in base_model.layers[:fine_tuning_at]:
  layer.trainable = False

# Compilando o modelo para fine tuning
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr = 0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# Fine tuning
model.fit_generator(train_generator, epochs=5, validation_data=valid_generator)

# Avaliação do modelo com fine tuning
valid_loss, valid_accuracy = model.evaluate_generator(valid_generator)
print(valid_accuracy)
