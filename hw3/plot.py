from PIL import Image
from matplotlib import pyplot as plt
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras import regularizers



from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import csv
import os

f = open('train.csv','r')

image = []

n_row= 0
for row in csv.reader(f):
	if(n_row==282):
		image.append(row[1].split())
	n_row = n_row+1
f.close()

"""--------------
load model
----------------"""
"""
json_file = open('model_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('model_1.h5')
print('Loaded model from disk')
"""
a = np.array(image)
a= a.astype('float32')


a = a.reshape(48,48)
plt.imshow(a, cmap='gray', interpolation='nearest', vmin=0, vmax=255)

#plt.imshow(image_array)
plt.savefig('sad.png')
plt.show()
