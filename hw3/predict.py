import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from keras.models import model_from_json
#from utils import *
import numpy as np
import pandas as pd
import csv
from sys import argv


def main():
	json_file = open('model_2.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights('model_2.h5')
	

	f = open(argv[1], 'r')
	x_t = []

	n_row = 0
	for row in csv.reader(f):
		if(n_row!=0):
			x_t.append(row[1].split())
		n_row = n_row+1

	x_test = np.array(x_t) # x.shape = 28709*2304 (2304 = 48*48)
	x_test = x_test.astype('float32')
	x_test = x_test/255

	x_test = x_test.reshape(x_test.shape[0],48,48,1)

	Y_test = loaded_model.predict(x_test)
	y_test = np.argmax(Y_test,axis=1)


	count = len(y_test)
	f = open(argv[2],'w')
	w = csv.writer(f)
	text = ['id','label']
	w.writerow(text)
#for i in range(result.shape[0]):
	for i in range(count):
		w.writerow([i,y_test[i]])
	f.close()
if __name__ == "__main__":
	main()
