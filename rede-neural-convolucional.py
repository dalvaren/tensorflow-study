import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10

# %matplotlib inline
print(tf.__version__)


# Configurando o nome das classes que serão previstas
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Carregando a base de dados
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalização
X_train = X_train / 255.0
X_test = X_test / 255.0

# plt.imshow(X_test[1])

# construção da rede neural
model = tf.keras.models.Sequential()
# camada de convolução
# filters (filtros): 32
# kernel_size (tamanho do kernel): 3
# padding (preenchimento): same
# função de ativação: relu
# input_shape (camada de entrada): (32, 32, 3)
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[32, 32, 3]))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
# max pooling
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# teceira camada de convolução
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# flattening
model.add(tf.keras.layers.Flatten())


# criada entrada rede neural
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))

# camada de saída
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
print(model.summary())

# compilando modelo
model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["sparse_categorical_accuracy"])

# treinando modelo
model.fit(X_train, y_train, epochs=15)

# avaliando modelo
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy: {}".format(test_accuracy))
test_loss
