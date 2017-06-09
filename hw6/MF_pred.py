import csv
import numpy as np
import keras.backend as K
from sys import argv
from keras.models import Sequential, Model
from keras.layers import Dense,Dropout,Input
from keras.layers import Flatten, Merge, Dot, Add, Concatenate, BatchNormalization
from keras.models import load_model
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split


n_user = 6041
n_movie = 3953
test_path = argv[1]
output_path = argv[2]

def main():
	latent_dim = 10

	user_input = Input(shape=[1])
	movie_input = Input(shape=[1])
	user_vec = Embedding(n_user, latent_dim)(user_input)
	user_vec = Flatten()(user_vec)
	movie_vec = Embedding(n_movie, latent_dim)(movie_input)
	movie_vec = Flatten()(movie_vec)
	#user_bias = Embedding(n_user, 1, embeddings_initializer='zeros')(user_input)
	#user_bias = Flatten()(user_bias)
	#movie_bias = Embedding(n_user, 1, embeddings_initializer='zeros')(movie_input)
	#movie_bias = Flatten()(movie_bias)
	r_hat = Dot(axes=1)([user_vec, movie_vec])
	#r_hat = Add()([r_hat, user_bias, movie_bias])
	model = Model(inputs = [user_input, movie_input], outputs=r_hat)
	model.load_weights('best_MF_10.hdf5')


	
	test_user = []
	test_age = []
	test_movie = []
	test_genre = []
	Y = []
	f = open(test_path+'test.csv','r')
	n_row = 0
	training_data = 0
	for line in f:
		if n_row>0:
			line = line.strip('\n')
			x = line.split(',')
			test_user.append(int(x[1]))
			test_movie.append(int(x[2]))

		n_row += 1
	f.close()

	test_user = np.array(test_user)
	test_movie = np.array(test_movie)


	Y_pred = model.predict([test_user,test_movie])


	count = Y_pred.shape[0]
	f = open(output_path,'w')
	w = csv.writer(f)
	text = ['TestDataID','Rating']
	w.writerow(text)
	#for i in range(result.shape[0]):
	for i in range(count):
		w.writerow([i+1,Y_pred[i][0]])
	f.close()





if __name__=='__main__':
	main()