import csv
import numpy as np
import keras.backend as K

from keras.models import Sequential, Model
from keras.layers import Dense,Input
from keras.layers import Flatten, Merge, Dot, Add
from keras.layers.embeddings import Embedding
from sklearn.cross_validation import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint


def MF_model(n_user,n_movie,latent_dim):
	user_input = Input(shape=[1])
	movie_input = Input(shape=[1])
	user_vec = Embedding(n_user, latent_dim, embeddings_initializer='random_normal')(user_input)
	user_vec = Flatten()(user_vec)
	movie_vec = Embedding(n_movie, latent_dim, embeddings_initializer='random_normal')(movie_input)
	movie_vec = Flatten()(movie_vec)
	#user_bias = Embedding(n_user, 1, embeddings_initializer='zeros')(user_input)
	#user_bias = Flatten()(user_bias)
	#movie_bias = Embedding(n_user, 1, embeddings_initializer='zeros')(movie_input)
	#movie_bias = Flatten()(movie_bias)
	r_hat = Dot(axes=1)([user_vec, movie_vec])
	#r_hat = Add()([r_hat, user_bias, movie_bias])
	model = Model(inputs = [user_input, movie_input], outputs=r_hat)
	return model





def main():
	userID = []
	f = open('users.csv','r')
	n_row = 0
	for line in f:
		if n_row>0:
			#start = line.find(':')
			#end = line.find('\"',start+1)
			line = line.strip('\n')
			tag = line.split('::')
			userID.append(int(tag[0]))
			
		n_row += 1
	f.close()
	userID = np.array(userID)
	n_user = np.amax(userID)+1

	movieID = []
	n_row = 0
	f = open('movies.csv','r',encoding='latin-1')
	for line in f:
		if n_row>0:
			line = line.strip('\n')
			tag = line.split('::')
			movieID.append(int(tag[0]))
		n_row += 1
	f.close()
	movieID = np.array(movieID)
	n_movie = np.amax(movieID)+1

	user = []
	movie = []
	Y = []
	f = open('train.csv','r')
	n_row = 0
	training_data = 0
	for line in f:
		if n_row>0:
			line = line.strip('\n')
			x = line.split(',')
			user.append([int(x[1])])
			movie.append([int(x[2])])
			Y.append([int(x[3])])
			training_data += 1
		n_row += 1
	f.close()

	X_user = np.array(user)
	X_movie = np.array(movie)
	Y_data = np.array(Y)
	#---------------------
	#--Rating Normalizaton-
	#---------------------
	

	
	print ('mean', mean)
	print ('std', std)

	a_userid, b_userid, a_movieid, b_movieid, a_y, b_y = train_test_split(X_user, X_movie, Y_data, test_size=0.1)
	print ('Training size:',a_userid.shape, a_movieid.shape, a_y.shape)
	a_y = np.reshape(a_y,(a_y.shape[0],1))
	b_y = np.reshape(b_y,(b_y.shape[0],1))

	#----------------
	#---- MODEL -----
	#----------------
	latent_dim = 20
	nb_epoch = 1000
	

	model = MF_model(n_user,n_movie,latent_dim)
	model.summary()
	model.compile(loss='mse', optimizer='adamax')
	earlystopping = EarlyStopping(monitor='val_loss', patience = 5, verbose=1, mode='min')
	checkpoint = ModelCheckpoint(filepath='MF_model.hdf5',
								 verbose=1,
								 save_best_only=True,
								 save_weights_only=True,
								 monitor='val_loss',
								 mode='min')

	hist = model.fit([a_userid, a_movieid] ,a_y, 
					 validation_data=([b_userid, b_movieid], b_y),
					 epochs=nb_epoch,
					 #batch_size = batch_size,
					 callbacks=[earlystopping,checkpoint])




if __name__=='__main__':
	main()