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

def sample_model(num_of_class):
    #initial model
    model = Sequential()
    #add dropout to reduce overfitting
    #model.add(Dropout(0.2, input_shape=(60, 160, 1)))
    #with 64 filters, 5*5 for convolutional kernel and activation 'relu'
    model.add(Conv2D(64, (5, 5), input_shape=(60, 160, 1), activation='relu'))
    #pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    #fully connected layer
    model.add(Dense(600, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(num_of_class, kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l1(0.01), 
                    activation='softmax'))
    return model

def get_data():
    #concatenate data
    path = os.getcwd()
    data = []
    for i in range(12):
        pData = []
        for j in range(10):
            if os.path.isfile(path+"/../data/generalData/part%d/%ddata.csv" %(i+1,j)): 
                tmp = pd.read_csv(path+"/../data/generalData/part%d/%ddata.csv" %(i+1,j))
                if len(tmp) > 100:
                    pData.append(tmp[:100])
                else:
                    pData.append(tmp)
            else:
                pData.append(None)
        data.append(pData) 
    return data

def preprocess_data(data, test_id):
    test = None
    train = None
    TrainSize = 0
    TestSize = 0
    # get the size for each expression
    tmpTrainSize, tmpTestSize = 0, 0
    for exp in range(10):
        tmpTrainSize = 0
        tmpTestSize = 0
        for j in range(12):
            if test_id == j and data[j][exp] is not None:
                tmpTestSize = len(data[j][exp])
            elif data[j][exp] is not None :
                tmpTrainSize += len(data[j][exp])
        if (TrainSize,TestSize) == (0,0):
            TrainSize = tmpTrainSize
            TestSize = tmpTestSize
        if TrainSize > tmpTrainSize and tmpTrainSize != 0:
            TrainSize = tmpTrainSize
        if TestSize > tmpTestSize and tmpTestSize != 0:
            TestSize = tmpTestSize
    #     print (TestSize, tmpTestSize)    
    # sys.exit(-1)
    for exp in range(10):
        expTrain = None
        for j in range(12):
            if test_id == j and data[j][exp] is not None:
                # concatenate testData
                ratio = float(TestSize)/len(data[j][exp])
                if ratio < 1:
                    discard, sTest = train_test_split(data[j][exp], test_size = ratio)
                    frames = [test,sTest]
                else:
                    frames = [test,data[j][exp]]
                test = pd.concat(frames)
            elif data[j][exp] is not None:
                # concatenate expTrain
                frames = [expTrain, data[j][exp]]
                expTrain = pd.concat(frames) 
        #concatenate train
        ratio = float(TrainSize)/len(expTrain)
        if ratio < 1:
            discard, sTrain  = train_test_split(expTrain, test_size = ratio) 
            frames = [train,sTrain]
        else:
            frames = [train,expTrain]
        train = pd.concat(frames)

    test_images = reshapeTo60and160(test)
    train_images = reshapeTo60and160(train)
    #reshape to [# of samples][width][height][pixels] for tensorflow-keras input format
    test_images = test_images.reshape(test_images.shape[0], 60, 160, 1).astype('float32')
    train_images = train_images.reshape(train_images.shape[0], 60, 160, 1).astype('float32')
    #normilize the data
    test_images = test_images/255
    train_images = train_images/255
    #One hot encode outputs: change target(emotion) values to input format with one-hot encode
    test_targets = np_utils.to_categorical(test.expression.values)
    train_targets = np_utils.to_categorical(train.expression.values)
    #set number of prediction classes
    num_classes = test_targets.shape[1]
    return train_images, train_targets, test_images, test_targets, test, num_classes

def train(model, train_data, train_label, test_data, test_label):

    model.summary()
    filename = path+"../data/generalData/general_Model.hdf5"
    check_point = ModelCheckpoint(filename, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
    callbacks_list = [check_point]
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_data, train_label, validation_data=(test_data, test_label), epochs=10, batch_size=80, callbacks=callbacks_list, verbose=2)

def test(model, test_data, test_label, test):

    scores = model.evaluate(test_data, test_label, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    #plot CM
    #print(test.expression.values)
    Saved_prediction = model.predict_classes(test_data, verbose=0)
    True_prediction = test.expression.values
    cm = confusion_matrix(True_prediction, Saved_prediction)

    class_names = ['blink', 'blink_left', 'blink_right', 'squint', 'frown', 'raise_eyebrow', 'enlarge',  'smile', 'smile_left', 'others']
    plot_confusion_matrix(cm, classes=class_names, normalize=True,
                          title='Confusion Matrix for Test Dataset')
    plt.show()
    return cvscores

if __name__ == '__main__': 
    print("get data")
    data = get_data()
    cvscores = []
    model = None
    for test_id in range(1):
        train_data, train_label, test_data, test_label, test, num_of_class = preprocess_data(data, test_id)
        model = sample_model(num_of_class)
        train(model, train_data, train_label, test_data, test_label)
        cvscores = test(model, test_data, test_label, test)
    
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))







