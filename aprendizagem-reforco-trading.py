import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas_datareader as data_reader
from pandas.util.testing import assert_frame_equal #import alterado

from tqdm import tqdm_notebook, tqdm
from collections import deque

print(tf.__version__)

class AI_Trader():
  
  def __init__(self, state_size, action_space = 3, model_name = "AITrader"):
    self.state_size = state_size #num de parâmetros camada de entrada
    self.action_space = action_space #num de parâmetros de saída
    self.memory = deque(maxlen = 2000)
    self.model_name = model_name
    
    self.gamma = 0.95 #fator de desconto
    self.epsilon = 1.0
    self.epsilon_final = 0.01
    self.epsilon_decay = 0.995
    self.model = self.model_builder()
    
  def model_builder(self):
    model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.LSTM(32,activation='tanh', recurrent_activation='elu',input_dim = (30,7)))  
    # model.add(tf.keras.layers.Dense(units = 64, activation = "relu"))
    # model.add(tf.keras.layers.Dense(units = 128, activation = "relu"))
    # model.add(tf.keras.layers.Dense(units = self.action_space, activation = "softmax"))
    # model.compile(loss = "mse", optimizer = tf.keras.optimizers.Adam(lr = 0.001))
    # model.add(tf.keras.Input(shape=(self.state_size,))) #input_dim é adicionado apenas à primeira camada, de entradas
    model.add(tf.keras.layers.Dense(units = 32, activation = "relu", input_dim = self.state_size))
    model.add(tf.keras.layers.Dense(units = 64, activation = "relu"))
    model.add(tf.keras.layers.Dense(units = 128, activation = "relu"))
    model.add(tf.keras.layers.Dense(units = self.action_space, activation = "linear"))
    model.compile(loss = "mse", optimizer = tf.keras.optimizers.Adam(lr = 0.001)) #loss function é o mean square error como visto para este modelo de aprendizagem
    return model

  
  def trade(self, state):
    if random.random() <= self.epsilon:
      return random.randrange(self.action_space)
    
    actions = self.model.predict(state) #retorna os valores de Q (possíveis valores de saída)
    return np.argmax(actions[0])
  
  def batch_train(self, batch_size):
    batch = []
    for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
      batch.append(self.memory[i])
      
    for state, action, reward, next_state, done in batch:
      if not done:
        # print(self.model.predict(next_state[0])[0])
        # print('---------')
        reward = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
        
      target = self.model.predict(state)
      target[0][action] = reward
      
      self.model.fit(state, target, epochs=1, verbose=0)
      
    if self.epsilon > self.epsilon_final:
      self.epsilon *= self.epsilon_decay

# Pré-processamento da base de dados

def sigmoid(x): #usado para a normalização dos preços entre 0 e 1... e neste caso apenas para isso
  return 1 / (1 + math.exp(-x))

def stocks_price_format(n):
  if n < 0:
    return "- $ {0:2f}".format(abs(n))
  else:
    return "$ {0:2f}".format(abs(n))

# Carregador da base de dados

dataset = data_reader.DataReader("AAPL", data_source = "yahoo")
dataset.head()

def dataset_loader(stock_name):
  dataset = data_reader.DataReader(stock_name, data_source = "yahoo")
  start_date = str(dataset.index[0]).split()[0]
  end_date = str(dataset.index[-1]).split()[0]
  close = dataset['Close']
  return close

def state_creator(data, timestep, window_size): #window_size = número de preços de trade a considerar
  starting_id = timestep - window_size + 1 #+1 por que o último não entra
  
  if starting_id >= 0:
    windowed_data = data[starting_id:timestep + 1]
  else:
    windowed_data = - starting_id * [data[0]] + list(data[0:timestep + 1])
    
  state = []
  for i in range(window_size - 1):
    state.append(sigmoid(windowed_data[i + 1] - windowed_data[i]))
    
  return np.array([state]), windowed_data

# Carregando a base de dados
stock_name = "AAPL"
data = dataset_loader(stock_name)

s, w = state_creator(data, 0, 5)

# Treinando a IA
window_size = 10
episodes = 1000
batch_size = 32
# batch_size = 1259 * 10
data_samples = len(data) - 1

# Definição do modelo
trader = AI_Trader(window_size)
trader.model.summary()
# Loop de treinamento
for episode in range(1, episodes + 1):
  print("Episode: {}/{}".format(episode, episodes))
  state = state_creator(data, 0, window_size + 1)
  total_profit = 0
  trader.inventory = []
  for t in tqdm(range(data_samples)):
    action = trader.trade(state)
    next_state = state_creator(data, t + 1, window_size + 1)
    reward = 0
    
    if action == 1: # Comprando uma ação
      trader.inventory.append(data[t])
      print("AI Trader bought: ", stocks_price_format(data[t]))
    elif action == 2 and len(trader.inventory) > 0: # Vendendo uma ação  
      buy_price = trader.inventory.pop(0)
      
      reward = max(data[t] - buy_price, 0)
      total_profit += data[t] - buy_price
      print("AI Trader sold: ", stocks_price_format(data[t]), " Profit: " + stocks_price_format(data[t] - buy_price))
      
    if t == data_samples - 1:
      done = True
    else:
      done = False
      
    trader.memory.append((state, action, reward, next_state, done))
    
    state = next_state
    
    if done:
      print("########################")
      print("Total profit: {}".format(total_profit))
      print("########################")
      
    if len(trader.memory) > batch_size:
      trader.batch_train(batch_size)
     
  if episode % 10 == 0:
    trader.model.save("ai_trader_{}.h5".format(episode))
    
