import sys, os, time
import logging

from scipy import misc
import numpy as np
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

def print_usage():
	print("Usage: python %s <data_dir>" % (__file__))

if __name__ == "__main__":

	logging.basicConfig(level=logging.DEBUG)

	if len(sys.argv) != 2:
		logging.error('bad arguments! (%d given)' % (len(sys.argv)))
		print_usage()
		sys.exit()
	if not os.path.isdir(sys.argv[1]):
		logging.error('bad arguments! (%s not a directory)' % (sys.argv[1]))
		print_usage()
		sys.exit()

	DATA_DIR = sys.argv[1]

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

	TRAIN_DATA = []
	TRAIN_LABEL = []

	for pNum in range(NUM_PARTICIPANT):
		for exNum in range(NUM_EXPRESSION):
			full_dir_path = os.path.join(DATA_DIR, "%s/%s" % (pNum, exNum))
			if not os.path.isdir(full_dir_path):
				logging.warning('expression %d not found for participant %d' % (exNum, pNum))
			else:
				for f in os.listdir(full_dir_path):
					if f.endswith('.jpg'):
						data_example = misc.imread(os.path.join(full_dir_path, f))
						assert type(data_example) == type(np.array([]))
						assert data_example.shape == (60, 160, 3)
						TRAIN_DATA.append(data_example)
						TRAIN_LABEL.append(exNum)

	logging.info("finished reading training data.")

	# ========== build the model ========== #
	logging.info("building the model...")

	model = Sequential()
	input_shape = (60, 160, 3)

	# 1) first convolution layer
	model.add(Conv2D(20, (5, 5), padding="same", input_shape=input_shape))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	# 2) second convolution layer
	model.add(Conv2D(50, (5, 5), padding="same"))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	# 3) fully connected layer
	model.add(Flatten())
	model.add(Dense(500))
	model.add(Activation("relu"))

	# 4) output layer (softmax classifier)
	model.add(Dense(NUM_EXPRESSION))
	model.add(Activation("softmax"))

	model.summary()

	# ========== training ========== #

	EPOCH = 25
	LEARNING_RATE = 1e-3
	BATCH_SIZE = 32


	logging.info("compiling model...")
	optimizer = Adam(lr=LEARNING_RATE)
	model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['categorical_accuracy'])

	logging.info("training started...")
	TRAIN_DATA = np.array(TRAIN_DATA)
	TRAIN_LABEL = np_utils.to_categorical(TRAIN_LABEL)
	model.fit(TRAIN_DATA, TRAIN_LABEL, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=2, validation_split=0.2, shuffle=False)

	logging.info("training completed.")
	model.save("main_model.%s.h5" % (str(time.time())))
