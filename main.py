import sys, os
import logging

from scipy import misc
import numpy as np

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
				logging.info('expression %d not found for participant %d' % (exNum, pNum))
			else:
				for f in os.listdir(full_dir_path):
					if f.endswith('.jpg'):
						data_example = misc.imread(os.path.join(full_dir_path, f))
						assert type(data_example) == type(np.array([]))
						assert data_example.shape == (60, 160, 3)
						TRAIN_DATA.append(data_example)
						TRAIN_LABEL.append(exNum)

	logging.info("finished reading training data.")
