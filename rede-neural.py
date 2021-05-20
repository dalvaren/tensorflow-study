import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

print(tf.__version__)

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print(X_train[0])

# normalização (transformar valores entre 0 e 1... 255 é o maior valor da base atual)
# o _test contem os resultados
X_train = X_train / 255.0
X_test = X_test / 255.0

# reshape
print(X_train.shape)
# Como a dimensão de cada imagem é 28x28, mudamos toda a base de dados para o formato [-1 (todos os elementos), altura * largura]
X_train = X_train.reshape(-1, 28*28)
print(X_train.shape)
# Mudamos também a dimensão da base de teste (precisa ser um vetor, pois cada entrada do vetor será um parâmetro da rede neural)
X_test = X_test.reshape(-1, 28*28)
print(X_test.shape)


# CONSTRUÇÃO DA REDE NEURAL
# definindo o modelo
model = tf.keras.models.Sequential()

# criando 1a camada oculta com 128 neurônios, função de ativação 'relu' e numero de parametros de entrada como 784
model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784, ))) 
# adição de mais camadas
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
# Dropout é uma técnica de regularização na qual alguns neurônios da camada tem seu valor mudado para zero, ou seja, durante o treinamento esses neurônios não serão atualizados. Com isso, temos menos chances de ocorrer overfitting
# 0.2 indica que 20% dos neurônios desta camada serão zerados
model.add(tf.keras.layers.Dropout(0.2))
# camada de saída (10 neurônios uma vez que existem 10 categorias de classificação das roupas... func de ativação como softmax por se tratar de classificação)
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

# compilando o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
print(model.summary())


# TREINAMENTO
model.fit(X_train, y_train, epochs=15)

# AVALIAÇÃO DO MODELO
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy: {}".format(test_accuracy))
print(test_loss)

# SALVANDO O MODELO
model_json = model.to_json()
with open("fashion_model.json", "w") as json_file:
    json_file.write(model_json)
# salvando os pesos
model.save_weights("fashion_model.h5")