# imports
import pandas as pd
import numpy as np
from numpy import load
from numpy import asarray
from numpy import save
from numpy import mean
from numpy import std
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from settings import LABELS_DIR, IMAGES_DIR, MODEL_H5, LE

'''
Prepare the raw data
Endode labels, save encoder
Convert pictures to arrays and save
Convert labels to arrays and save
Create functions:
1- prep_data() 
Load arrays, split into train/test, encode y, normalize values of X
2- define_model()
''' 

# load data
data = pd.read_csv(LABELS_DIR + "labels.csv", index_col=0)

# create labels column and encode
le = LabelEncoder()
data['labels'] = le.fit_transform(data['chord_instructed'])

# save label encoder
import pickle
pickle.dump(le, open(LE, 'wb'))
classes = len(data['labels'].unique())

# load photos and convert to numpy array
print("Converting photos to np array")
photos = list()
for name in data.id.values:
    # load image
    photo = load_img(IMAGES_DIR + name + '.png', color_mode = "grayscale", target_size=(64,64))
     # convert to np array
    photo = img_to_array(photo)
    #store
    photos.append(photo)
labels = list(data['labels'].values)
photos = asarray(photos)
labels = asarray(labels)

# save arrays
save(LABELS_DIR + 'photos.npy', photos)
save(LABELS_DIR + 'labels.npy', labels)
print("Finished")

# define functions
def prep_data():
    # load arrays
    photos = load(LABELS_DIR + 'photos.npy')
    labels = load(LABELS_DIR + 'labels.npy')
    
    # split between train/test and reshape
    X_train, X_test, y_train, y_test = train_test_split(photos, labels, test_size=0.15, shuffle=False)
    y_train = np.reshape(y_train, (X_train.shape[0],))
    y_test = np.reshape(y_test, (X_test.shape[0],))
    
    # hot encode target variables
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    # normalize dependant variables to range 0-1
    X_train = X_train/255.0
    X_test = X_test/255.0
    return X_train, X_test, y_train, y_test

def define_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu',  padding='same', kernel_initializer='he_uniform', input_shape=(64, 64, 1))) #conv layer
    model.add(layers.MaxPooling2D((2,2))) #pooling layer
    model.add(layers.Conv2D(64, (3,3), activation='relu',  padding='same', kernel_initializer='he_uniform'))
    model.add(layers.MaxPooling2D((2,2))) #pooling layer
    model.add(layers.Conv2D(64, (3,3), activation='relu',  padding='same', kernel_initializer='he_uniform'))              
    model.add(layers.Flatten()) # 16384 vector
    model.add(layers.Dense(100, activation='relu', kernel_initializer='he_uniform')) # 1 hidden layer
    model.add(layers.Dense(classes, activation='softmax')) # output, size 42 = nuber of classes
    # compile model
    opt = optimizers.SGD(lr=0.01, momentum=0.9) # gradient descent with momentum optimizer
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# train model
X_train, X_test, y_train, y_test = prep_data()
model = define_model()
print('Fitting model')
model.fit(X_train, y_train, epochs=30, batch_size=24)
print('Accuracy')
_, acc = model.evaluate(X_test, y_test, verbose=0)
print('> %.3f' % (acc * 100.0))

# save model
model.save(MODEL_H5)