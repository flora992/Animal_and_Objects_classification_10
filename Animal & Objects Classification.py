

!pip install --upgrade --force-reinstall --no-deps kaggle
!kaggle --version

from google.colab import files
files.upload()

!ls -lha kaggle.json

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json

!kaggle competitions download -c codeml-challenge5 --force

from zipfile import ZipFile
file_name = 'codeml-challenge5.zip'

with ZipFile(file_name, 'r') as zip:
  zip.extractall()
  print('Done')

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import cv2
from PIL import Image
import keras
from natsort import natsorted, ns
from keras.utils import to_categorical 
from keras.utils import normalize
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
import pandas as pd

# set your own dir for trainset and testset
train_dir = './train_images/train_images/' 
test_dir = './test_images/content/data/test_imagess/'

# One Hot Encode training data labels
train_labels = np.genfromtxt("./train_label.txt",delimiter='\n',dtype=None,encoding=None)



# Encode target labels with beluw between 0 and n_classes-1. 

label_encoder = LabelEncoder() # Create an instance

# Fit label encoder (One-Hot Encode) the labels as a vector
vector = label_encoder.fit_transform(train_labels)

#Convert to categorical
y_train = to_categorical(vector) #Shape: (49999, 10)

print(y_train.shape)
print(y_train[0])
train_labels[:10] #same as the original txt file


SIZE = 32
x_train = []
x_test = []

# Output should be two lists of 50000 32x32 images with each pixel being an RGB value. 

for image_name in natsorted(os.listdir(train_dir)):
  if (len(image_name) > 3):
    if (image_name.split('.')[1] == 'png'):
      image = cv2.imread(train_dir + image_name)
      image = Image.fromarray(image, 'RGB')
      image = image.resize((SIZE, SIZE))
      x_train.append(np.array(image))

print('Loaded train set')

for image_name in natsorted(os.listdir(test_dir)):
  if (len(image_name) > 3):
    if (image_name.split('.')[1] == 'png'):
      image = cv2.imread(test_dir + image_name)
      image = Image.fromarray(image, 'RGB')
      image = image.resize((SIZE, SIZE))
      x_test.append(np.array(image))

print('Loaded test set')


X_train = normalize(x_train, axis = 1)
X_test = normalize(x_test, axis = 1)

print(X_train.shape)
print(X_test.shape)
print(y_train[0].shape)


Xtrain, Xvalidation, Ytrain, Yvalidation = train_test_split(X_train, y_train, test_size=9999, random_state = 0)

print(Xtrain.shape)
print(Ytrain.shape)
print(Xvalidation.shape)
print(Yvalidation.shape)

plt.imshow(Xvalidation[199])
print(Yvalidation[199])


model = keras.models.Sequential()
model.add(keras.layers.Conv2D(128, (3,3), padding='same', activation='relu',input_shape=(32,32,3)))
model.add(keras.layers.Conv2D(128, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Dropout(0.6))

model.add(keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Dropout(0.6))

#model.add(keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'))
#model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
#model.add(keras.layers.MaxPooling2D(2,2))
#model.add(keras.layers.Dropout(0.6))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(64, activation='relu'))
#model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))


model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(Xtrain, Ytrain, batch_size = 2048, epochs=20, verbose=1, validation_data=(Xvalidation, Yvalidation))

model.evaluate(Xvalidation, Yvalidation, verbose=0)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.plot(epochs, loss, 'c', label='Training loss')
plt.plot(epochs, val_loss, 'm', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend(loc=0)
plt.figure()
plt.show()


# Predict classes of X_test images (one-hot encoded format)
test_pred = model.predict(X_test)
test_pred_labels = [np.argmax(r) for r in test_pred]

# Generate data with specified format and convert test_pred to int
data = {'id':       list(range(0, len(X_test), 1)),
        'classes':  test_pred_labels}

# Generate the dataframe
df = pd.DataFrame(data, columns = ['id', 'classes'])

# Export df to csv
df.to_csv("./prediction_submission.csv", index = False)

print("finished")

