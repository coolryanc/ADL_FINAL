import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, BatchNormalization, Convolution2D, MaxPooling2D, Flatten, LeakyReLU
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import EarlyStopping
import utils
# from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import sys, os
import sys, os
import argparse
import logging
from scipy import misc
import random

def parse():
	parser = argparse.ArgumentParser(description = "CNN Network")
	parser.add_argument('--epochs', default = 6, type = int)
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--lr', default=0.0005, type=float,
						help="Initial learning rate")
	parser.add_argument('--save_dir', default='./model')
	parser.add_argument('-t', '--testing', action='store_true',
						help="Test the trained model on testing dataset")
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse()
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	DATA_DIR = './data/'

	# ========== get the images (raw data) ========= #
	#
	# 0_exp: blink both      => 0
	# 1_exp: blink left      => 1
	# 2_exp: blink right     => 2
	# 3_exp: squint both     => 3
	# 5_exp: raise eyebrows  => 4
	# 6_exp: enlarge         => 5
	#

	# some global settings
	NUM_EXPRESSION = 6
	NUM_PARTICIPANT = 12

	# get training data
	logging.info("reading training data...")

	X = []
	Y = []

	for pNum in range(NUM_PARTICIPANT):
		for exNum in range(NUM_EXPRESSION):
			full_dir_path = os.path.join(DATA_DIR, "%s/%s" % (pNum, exNum))
			if not os.path.isdir(full_dir_path):
				logging.info('expression %d not found for participant %d' % (exNum, pNum))
			else:
				for f in os.listdir(full_dir_path):
					if f.endswith('.jpg'):
						data_example = misc.imread(os.path.join(full_dir_path, f))
						assert type(data_example) == type(np.array([]))
						assert data_example.shape == (60, 160, 3)
						X.append(data_example / 255)
						Y.append(exNum)

	logging.info("finished reading training data.")

	s = list(zip(X, Y))
	random.shuffle(s)

	X, Y = zip(*s)

	X = np.array(X)
	Y = np.array(Y)

	print(X.shape)
	print(Y.shape)

	X_train, X_valid = np.array(X[:-500]), np.array(X[-500:])
	Y_train, Y_valid = np.array(Y[:-500]), np.array(Y[-500:])

	input_shape = (60, 160, 3)
	num_classes = 6

	Y_train = np_utils.to_categorical(Y_train, num_classes=num_classes)
	Y_valid = np_utils.to_categorical(Y_valid, num_classes=num_classes)

	# train_data_gen = ImageDataGenerator(
	# 					rotation_range=30,
	# 					width_shift_range=0.2,
	# 					height_shift_range=0.2,
	# 					zoom_range=[0.8, 1.2],
	# 					shear_range=0.2,
	# 					horizontal_flip=True)

	model = Sequential()
	model.add(Convolution2D(64, (3, 3), padding='same',
					 input_shape=input_shape))
	model.add(LeakyReLU(alpha=0.03))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.2))

	model.add(Convolution2D(128, (3, 3), padding='same'))
	model.add(LeakyReLU(alpha=0.03))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.25))

	model.add(Convolution2D(512, (3, 3), padding='same'))
	model.add(LeakyReLU(alpha=0.03))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.3))

	model.add(Convolution2D(512, (3, 3), padding='same'))
	model.add(LeakyReLU(alpha=0.03))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.35))

	model.add(Flatten())

	model.add(Dense(512, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(512, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	optimizer = Adam(lr=0.0005)
	model.compile(loss='categorical_crossentropy',
				  optimizer= optimizer,
				  metrics=['accuracy'])
	model.summary()

	# train_history = model.fit_generator(
	# 		train_data_gen.flow(X_train, Y_train, batch_size=args.batch_size),
	# 		steps_per_epoch=5*len(X_train)//args.batch_size,
	# 		epochs=args.epochs,
	# 		validation_data=(X_valid, Y_valid)
	# 		)

	callbacks = []
	callbacks.append(ModelCheckpoint('./model/model-{epoch:05d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=True, period=1))

	model.fit(X_train, Y_train, batch_size=args.batch_size, epochs=args.epochs, validation_data=(X_valid, Y_valid))

	# model.fit_generator(
	# 		train_data_gen.flow(X_train, Y_train, batch_size=args.batch_size),
	# 		steps_per_epoch=5*len(X_train)//args.batch_size,
	# 		epochs=args.epochs,
	# 		validation_data=(X_valid, Y_valid),
	# 		callbacks=callbacks
	# 		)

	model.save(args.save_dir + '/cnn.h5')

	score = model.evaluate(X_valid, Y_valid)
	print("Loss: {}".format(score[0]))
	print("Accuracy: {}".format(score[1]))
	#
	y_pred = model.predict_classes(X_valid)
	class_names = ['blink both ', 'blink left', 'blink right', 'squint both', 'raise eyebrows', 'enlarge ']
	cnf_matrix = confusion_matrix(np.argmax(Y_valid,axis=1), y_pred)
	utils.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
					  title='Normalized confusion matrix')

	# plot_model(model, to_file='cnn_model.png')
	# model.save('./model_cnn_v9.h5')
