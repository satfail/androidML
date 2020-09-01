# -*- coding: utf-8 -*-



import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


#Carga de dataset mnist
mnist = tf.keras.datasets.mnist
(X_train, y_train),(X_test, y_test) = mnist.load_data()

#Labels
class_names = ["zero","one","two","three","four","five","six","seven","eight","nine"]

#Plot de una imagen
plt.figure()
plt.imshow(X_test[0])
plt.colorbar()
plt.grid(False)
plt.xlabel("Classification label: {}".format(y_train[0]))
plt.show()

#Representación de los valores rgb en el dominio [0,1] 
X_train = X_train / 255.0
X_test = X_test / 255.0

# Creamos con Keras el modelo secuencual
  # 1º Aplanamos la primera capa de 2D a 1D 
  # 2º Capa con activación relu g(z) = max{0, z}	para una activación casi lineal
   # https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
  # 3º Creamos un Dropout del 0.2 de las neuronas de la capa, para evitar el overfitting
    # https://dl.acm.org/doi/pdf/10.5555/2627435.2670313
  # 4º Creamos la capa de salida con 10 nodos uno para cada lable, la función softmax nos dará la probabilidad de cada elemento
    # https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

#Compilamos el modelo
  # Usamos el algoritmo de optimización adam
    #https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
  # Utilizamos sparse, para evitar tener hacer onehotencode de los targets
    # https://www.dlology.com/blog/how-to-use-keras-sparse_categorical_crossentropy/
  # accuracy, frecuencia que predictions = y_test
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamos
model.fit(X_train, y_train, epochs=20)
# Métricas
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Predicción
predictions = model.predict(X_test)

#Creación de modelo tflitle para usarlo en android
kearas_file = "digit.h5"
tf.keras.models.save_model(model,kearas_file)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()
open("digit.tflite","wb").write(tfmodel)
