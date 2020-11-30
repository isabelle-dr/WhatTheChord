# imports
import pandas as pd
import numpy as np
from numpy import load
from numpy import asarray
from numpy import save
from numpy import mean
from numpy import std
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

path_images = "images/"
data = pd.read_csv("labels/chords.csv", index_col=0)
data=data.dropna()

############# PREP DATA ################
# create labels column and encode
le = LabelEncoder()
data['labels'] = le.fit_transform(data['chord_instructed'])

# save label encoder
import pickle
filename = 'labelencoder.sav'
pickle.dump(le, open(filename, 'wb'))

# load photos and convert to numpy array
print("Converting photos to np array")
photos = list()
for name in data.id.values:
    # load image
    photo = load_img(path_images + name + '.png', color_mode = "grayscale", target_size=(64,64))
     # convert to np array
    photo = img_to_array(photo)
    #store
    photos.append(photo)
labels = list(data['labels'].values)
photos = asarray(photos)
labels = asarray(labels)

# save arrays
save('photos.npy', photos)
save('labels.npy', labels)

############# DEFINE FUNCTIONS ################
def load_dataset():
    # load arrays
    photos = load('photos.npy')
    labels = load('labels.npy')
    
    # split between train/test and reshape
    X_train, X_test, y_train, y_test = train_test_split(photos, labels, test_size=0.15, shuffle=False)
    y_train = np.reshape(y_train, (X_train.shape[0],))
    y_test = np.reshape(y_test, (X_test.shape[0],))
    
    # hot encode target variables
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return X_train, X_test, y_train, y_test

def prep_pixels(train, test):
    # convert from integers to float
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm/255.0
    test_norm = test_norm/255.0
    # return normalized images
    return train_norm, test_norm

def define_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu',  padding='same', kernel_initializer='he_uniform', input_shape=(64, 64, 1))) #conv layer
    model.add(layers.MaxPooling2D((2,2))) #pooling layer
    model.add(layers.Conv2D(64, (3,3), activation='relu',  padding='same', kernel_initializer='he_uniform'))
    model.add(layers.MaxPooling2D((2,2))) #pooling layer
    model.add(layers.Conv2D(64, (3,3), activation='relu',  padding='same', kernel_initializer='he_uniform'))              
    model.add(layers.Flatten()) # 16384 vector
    model.add(layers.Dense(100, activation='relu', kernel_initializer='he_uniform')) # 1 hidden layer
    model.add(layers.Dense(42, activation='softmax')) # output, size 42 = nuber of classes
    # compile model
    opt = optimizers.SGD(lr=0.01, momentum=0.9) # gradient descent with momentum optimizer
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        plt.subplot(211)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        plt.subplot(212)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    plt.show()

def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    # box and whisker plots of results
    plt.boxplot(scores)
    plt.show()


############# CALL FUNCTIONS AND FIT MODEL ################
# load dataset
X_train, X_test, y_train, y_test = load_dataset()
# prepare pixel data
X_train, X_test = prep_pixels(X_train, X_test)
model = define_model()
print('Fitting model')
model.fit(X_train, y_train, epochs=20, batch_size=32)
print('Accuracy')
_, acc = model.evaluate(X_test, y_test, verbose=0)
print('> %.3f' % (acc * 100.0))

# save model
model.save('final_model.h5')
