import numpy as np
from keras.models import load_model

model_path = "data/generalData/general_Model.hdf5"

test_data_images_path = "test_data_images.nparray"
test_data_targets_path = "test_data_targets.nparray"

model = load_model(model_path)

test_images = np.load(test_data_images_path)
test_targets = np.load(test_data_targets_path)
scores = model.evaluate(test_images, test_targets, verbose=0)

print("accuracy:", scores[1])
