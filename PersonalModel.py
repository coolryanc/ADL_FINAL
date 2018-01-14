import pandas as pd
import numpy as np
import os
import sys
import os.path
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D 
from keras.utils import np_utils, plot_model
#from keras import backend as K
# from keras.regularizers import L1L2, l1
# activity_l1 = L1L2(l1=1)
from keras import regularizers
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
import itertools

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, int(cm[i, j]*100)/100.0,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#convert flatten data to 48*48 matrix
def reshapeTo60and160(dataset):
    #extract pixels value from original pandas dataframe
    pixels_values = dataset.pixels.str.split(" ").tolist()
    #convert pixels of each image to 48*48 formats
    images = []
    for image in np.array(pixels_values, dtype=float):
        images.append(image.reshape(60, 160))
    return np.array(images, dtype=float)

def sample_model():
    #initial model
    model = Sequential()
    #add dropout to reduce overfitting
    #model.add(Dropout(0.2, input_shape=(60, 160, 1)))
    #with 64 filters, 5*5 for convolutional kernel and activation 'relu'
    model.add(Conv2D(64, (5, 5), input_shape=(60, 160, 1), activation='relu'))
    #pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    #fully connected layer
    model.add(Dense(600, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(num_classes, kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l1(0.01), 
                    activation='softmax'))
    return model
#concatenate data
trainList = []
test = None
path = os.getcwd()
data = []
size = 0
dataSize = 0
name = sys.argv[1]
for subdir, dirs, files in os.walk(path):
    if subdir.endswith(name):
        for file in files:
            if file.endswith("data.csv"):
                dataSize += 1
                tmpData = pd.read_csv(subdir+"/"+file)
            # os.path.isfile(path+"/data/"+name"/%ddata.csv" %i):
            # tmpData = pd.read_csv(path+"/data/3_Tey_Yin_Cheng/%ddata.csv" %i)
                if size == 0:
                    size = len(tmpData)
                elif len(tmpData) < size:
                    print("len:",len(tmpData))
                    size = len(tmpData)
                print("size:",size)
                data.append(tmpData)

for i in range(len(data)):
    ratio = float(size)/len(data[i])
    if ratio != 1:
        exData, sData = train_test_split(data[i], test_size = ratio)
        train, tmp_test = train_test_split(sData, test_size = 0.2)
        #train, tmp_test = train_test_split(data[i], test_size = 0.2)
    else: 
        train, tmp_test = train_test_split(data[i], test_size = 0.2)
    #split each expression into k-fold data
    trainSplit = []
    k = 10
    for j in range(k):
        if j == k-1:
            trainSplit.append(train)
            break
        rTrain, sTrain = train_test_split(train, test_size = 1./(k-j))
        trainSplit.append(sTrain)
        train = rTrain
    trainList.append(trainSplit)
    frames = [test, tmp_test]
    test = pd.concat(frames)
test_images = reshapeTo60and160(test)

#reshape to [# of samples][width][height][pixels] for tensorflow-keras input format
#train_images = train_images.reshape(train_images.shape[0], 60, 160, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 60, 160, 1).astype('float32')

#check input format
# train_images.shape

#normilize the data
#train_images = train_images/255
test_images = test_images/255.

#One hot encode outputs: change target(emotion) values to input format with one-hot encode
#train_targets = np_utils.to_categorical(train.expression.values)
test_targets = np_utils.to_categorical(test.expression.values)

#set number of prediction classes
num_classes = test_targets.shape[1]
#10-fold cross validation
cvscores = []
model = None
for i in range(1):
    #concatenate 9 train and 1 validation
    train = None
    val = None
    for exp in range(dataSize): 
        strain = None 
        sval = None  
        for j in range(k):
            if j == i:
                sval = trainList[exp][j]
                continue
            frames = [strain, trainList[exp][j]]
            strain = pd.concat(frames)
        frames = [train, strain]
        train = pd.concat(frames)
        frames = [val, sval]
        val = pd.concat(frames)

    val_images = reshapeTo60and160(val)
    train_images = reshapeTo60and160(train)
    #reshape to [# of samples][width][height][pixels] for tensorflow-keras input format
    val_images = val_images.reshape(val_images.shape[0], 60, 160, 1).astype('float32')
    train_images = train_images.reshape(train_images.shape[0], 60, 160, 1).astype('float32')
    #normilize the data
    val_images = val_images/255
    train_images = train_images/255
    #One hot encode outputs: change target(emotion) values to input format with one-hot encode
    val_targets = np_utils.to_categorical(val.expression.values)
    train_targets = np_utils.to_categorical(train.expression.values)

    model = sample_model()
    # plot_model(model, to_file='model.png')
    # model.summary()
    # sys.exit(-1)
    filename = path+"/data/"+name+"/"+name+".hdf5"
    check_point = ModelCheckpoint(filename, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
    callbacks_list = [check_point]
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_images, train_targets, validation_data=(val_images, val_targets), 
          epochs=7, batch_size=80, callbacks=callbacks_list, verbose=2)
    scores = model.evaluate(test_images, test_targets, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    print(i)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

#plot CM
# print(test.expression.values)
# Saved_prediction = model.predict_classes(test_images, verbose=0)
# True_prediction = test.expression.values
# cm = confusion_matrix(True_prediction, Saved_prediction)

# class_names = ['blink_both', 'blink_left', 'blink_right', 'squint', 'squint_left', 'squint_right', 'frown', 'raise_eyebrow', 'enlarge',  'smile_both', 'others']
# plot_confusion_matrix(cm, classes=class_names, normalize=True,
#                       title='Confusion Matrix for Test Dataset')
# plt.show()




