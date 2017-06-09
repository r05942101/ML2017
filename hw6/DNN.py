import csv
import numpy as np
import keras.backend as K

from keras.models import Model
from keras.layers import Dense,Input,Concatenate
from keras.layers import Flatten, Merge, Dot, Add, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.layers.embeddings import Embedding
from sklearn.cross_validation import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint



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
	#Y_data = to_categorical(Y_data,5)
	#print (Y_data[:10])
	#---------------------
	#--Rating Normalizaton-
	#---------------------
	"""

	"""
	print ('n_user',n_user)
	print ('n_movie',n_movie)

	a_userid, b_userid, a_movieid, b_movieid, a_y, b_y = train_test_split(X_user, X_movie, Y_data, test_size=0.1)
	print ('Training size:',a_userid.shape, a_movieid.shape, a_y.shape)


	#----------------
	#---- MODEL -----
	#----------------
	latent_dim = 200
	nb_epoch = 1000
	#batch_size = 64
	

	user_input = Input(shape=[1])
	movie_input = Input(shape=[1])
	user_vec = Embedding(n_user, latent_dim, embeddings_initializer='random_normal')(user_input)
	#user_vec = Dropout(0.5)(user_vec)
	user_vec = Flatten()(user_vec)
	movie_vec = Embedding(n_movie, latent_dim, embeddings_initializer='random_normal')(movie_input)
	#movie_vec = Dropout(0.5)(movie_vec)
	movie_vec = Flatten()(movie_vec)
	
	merge_vec = Concatenate()([user_vec, movie_vec])
	#hidden = Dense(128,activation='relu')(merge_vec)
	#hidden = Dropout(0.1)(hidden)
	#hidden = BatchNormalization()(hidden)
	hidden = Dropout(0.1)(merge_vec)
	hidden = Dense(200,activation='relu')(hidden)
	hidden = Dropout(0.1)(hidden)
	#hidden = BatchNormalization()(hidden)
	#hidden = Dense(128,activation='relu')(hidden)
	#hidden = Dropout(0.5)(hidden)
	output = Dense(1)(hidden)

	model = Model(inputs=[user_input, movie_input], outputs=output)
	model.summary()
	#model.compile(loss='mse', optimizer='adamax')
	model.compile(loss='mse', optimizer='adamax')
	earlystopping = EarlyStopping(monitor='val_loss', patience = 6, verbose=1, mode='min')
	checkpoint = ModelCheckpoint(filepath='DNN.hdf5',
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

	#user_embeddings = np.array(model.layers[2].get_weights()).squeeze()
	#movie_embeddings = np.array(model.layers[3].get_weights()).squeeze()
	#print (user_embeddings.shape)
	#np.save('user_emb.npy',user_embeddings)
	#np.save('movie_emb.npy',movie_embeddings)
	

if __name__=='__main__':
	main()