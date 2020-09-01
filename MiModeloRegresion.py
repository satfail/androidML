# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:34:02 2020

@author: Miguel Ángel Gragera García
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importar el data set
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# Codificar datos categóricos
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Crea un vector para cada label 
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],   
    remainder='passthrough'                        
)
X = np.array(ct.fit_transform(X), dtype=np.float)

# Hay que evitar la trampa de las dummies quitar 1
X = X[:, 1:]

#Dividir el Dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Regresión lineal múltiple
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
X_train.shape
def build_model():
  model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(5,)),
    layers.Dense(10, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

model.summary()


EPOCHS = 1000

history = model.fit(
  X_train, y_train,
  epochs=EPOCHS)

#Veáse el [0] si no se pone nos da un shape(, 5) y no un (1, 5) como se necesita
X_test[[0], :].shape
y_pred = model.predict(X_test)


kearas_file = "startUps.h5"
tf.keras.models.save_model(model,kearas_file)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()
open("startUps.tflite","wb").write(tfmodel)

