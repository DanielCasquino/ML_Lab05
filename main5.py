import numpy as np
import math
import pandas as pd
import random
import matplotlib.pyplot as plt
import cvxopt
import seaborn as sns
import itertools
import pywt
import os
from sklearn.metrics import confusion_matrix
import zipfile
from skimage import io, color

# Data Normalizing
def normalizacion(data):
  return data
 # write your code here


# Wavelet function for extracting image characteristics. This function returns a vector with image characteristics
def haar(f, n):
  imagen = io.imread(f)
  for i in range(n):
    imagen, (LH, HL, HH) = pywt.dwt2(imagen, 'haar')
  gfg = np.array(imagen)
  return gfg.flatten()

# This function loads all images in the specified path and returns a vector containing the feature vectors of these images. In the example, we use three splits.
def encode(path, cuts):
  encodings = []

  print("Ingresó : ")
  print(path)
  i=0
  for filename in os.listdir(path):
      image_file = os.path.join(path, filename)
      print(image_file)
      if os.path.isfile(image_file):
          encodings.append(haar(image_file, cuts))
      i=i+1
      if i==20:
        return encodings
  return encodings


def get_data(emotion_1, emotion_2, cuts):
  data  =  np.array(encode(emotion_1,cuts))
  data  = np.insert(data, 0, 1, axis=1)
  temp =  np.array(encode(emotion_2,cuts))
  temp = np.insert(temp, 0, -1, axis=1)
  data = np.concatenate((data, temp), axis=0)
  for i in range(10):
    np.random.shuffle(data)
  y = data[:,0]
  x = data[:, 1:]
  return x, y

c1_emotion = 'fear'
c2_emotion = 'neutral'

c1_emotion_train  = './train/' + c1_emotion
c2_emotion_train  = './train/' + c2_emotion

c1_emotion_test  = './validation/' + c1_emotion
c2_emotion_test  = './validation/' + c2_emotion

train_x, train_y = get_data(c1_emotion_train, c2_emotion_train,1)
test_x, test_y = get_data(c1_emotion_test, c2_emotion_test,1)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

# Hypothesis
def h(X,w,b):
  return np.dot(X, w.T) + b

# Implemente la función de pérdida del soft SVM
def loss(y, y_aprox, W, C):
  # y is a 40 array
  # y_aprox is a 40 array
  # W is a 40 array
  # C is a constant
  l = []
  for i in range(y.shape[0]):
    l.append(1/2 * np.linalg.norm(W[i])**2 + C*np.sum(np.max(0.0, 1.0 - y[i] * y_aprox[i])))
  return np.array(l)

# Implemente la función para obtener las derivadas de W
def derivatives(x, y,y_aprox, w, b, C):
  # x is a 40, 576 matrix
  # y is a 40 vector
  # y_aprox is a 40 vector
  dw = []
  db = []
  for i in range(y.shape[0]):
    if(y[i] * y_aprox[i] < 1):
      dw.append(w + C * np.dot(-y, x))
      db.append(C*(-y[i]))
  else:
      dw.append(w)
      db.append(0)
  return np.array(dw), np.array(db)

def Update(x,y,y_aprox,w, b, db, dw, alpha, C):
 C = 1
 new_w = []
 new_b = []
 for i in range(y.shape[0]):
  if y[i] * y_aprox[i] < 1:
    new_w.append(w[i] - alpha*(C * np.sum(np.dot(-y[i], x[i]))))
  else:
     new_w.append(w[i] - alpha * w[i])
  return np.array(new_w), np.array(new_b)

# Training
def training(X, Y, C, alpha, epochs):
  w = np.array([np.random.rand() for i in range(X.shape[1])])
  b = np.random.rand()
  error = []
  for i in range(epochs):
    Y_aprox = h(X,w,b)
    dw, db = derivatives(X, Y, Y_aprox, w, b, C)
    w, b = Update(X,Y,Y_aprox,w, b, db, dw, alpha, C)
    L = loss(Y, Y_aprox, w, C)
    error.append(L)
  return w, b, error

# Testing

# Implemente la función de testing
def testing(X,W,b):
  y_aprox = []
  # write your code here
  for i in range(X.shape[0]):
    y_aprox = np.sign(np.dot(X[i], W.transpose())+b)
  return np.array(y_aprox)

# Main Program

m = train_y.size
k = train_x[0].size
train_x_norm = np.apply_along_axis(normalizacion, 1, train_x)
test_x_norm = np.apply_along_axis(normalizacion, 1, test_x)

W, b, e1, = training(train_x_norm, train_y, 1e6, 1e-8, 1200)
m = test_y.size
y_pred = testing(test_x_norm, W, b)
test_y = test_y.astype('int')

correct = 1

print("Clasificados correctamente:", correct)
print("Clasificados incorrectamente:", len(test_y)-correct)
print("% de efectividad", round(100*correct/len(test_y), 2))

matrix = confusion_matrix(test_y,y_pred)
df2 = pd.DataFrame(matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis], index=["Angry", 'Happy'], columns=["Angry", 'Happy'])
sns.heatmap(df2, annot=True, cbar=None, cmap="Greens")
plt.title("Confusion Matrix"), plt.tight_layout()
plt.xlabel("Predicted")
plt.ylabel("Real")
plt.show()

"""[Documento de excel](https://docs.google.com/spreadsheets/d/1yxYCjj_uS2Wrj8ofAc7hiRoCYX0-wJNiTMzGnWdsZoA/edit?usp=sharing)"""