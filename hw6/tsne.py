import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

import keras.backend as K

from keras.models import Sequential, Model
from keras.layers import Dense,Dropout,Input
from keras.layers import Flatten, Merge, Dot, Add, Concatenate, BatchNormalization
from keras.models import load_model
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split



n_user = 6041
n_movie = 3953

def draw(x,y):
	y = np.array(y)
	x = np.array(x,dtype=np.float64)
	model1 = TSNE(n_components=2)
	np.set_printoptions(suppress=True)
	vis_data = model1.fit_transform(x) 
	
	vis_x = vis_data[:,0]
	vis_y = vis_data[:,1]
	cm = plt.cm.get_cmap('RdYlBu')
	sc = plt.scatter(vis_x,vis_y, c=y, cmap=cm)#, cmap=plt.cm.get_cmap("jet", 10))
	plt.colorbar(sc)
	plt.show()


def main():

	latent_dim = 5
	user_input = Input(shape=[1])
	user_age_input = Input(shape=[1])
	movie_input = Input(shape=[1])
	movie_type_input = Input(shape=[1])
	user_vec = Embedding(n_user, latent_dim, embeddings_initializer='random_normal')(user_input)
	user_vec = Flatten()(user_vec)

	user_age_vec = Embedding(n_age, latent_dim, embeddings_initializer='random_normal')(user_age_input)
	user_age_vec = Flatten()(user_age_vec)
	
	movie_vec = Embedding(n_movie, latent_dim, embeddings_initializer='random_normal')(movie_input)
	movie_vec = Flatten()(movie_vec)

	movie_type_vec = Embedding(n_type, latent_dim, embeddings_initializer='random_normal')(movie_type_input)
	movie_type_vec = Flatten()(movie_type_vec)

	merge_user_vec = Concatenate()([user_vec, user_age_vec])
	merge_movie_vec = Concatenate()([movie_vec, movie_type_vec])

	r_hat = Dot(axes=1)([merge_user_vec, merge_movie_vec])
	model = Model(inputs = [user_input,user_age_input, movie_input, movie_type_input],
				  outputs=r_hat)
	model.load_weights('bonus.hdf5')

	movie_emb = np.array(model.layers[3].get_weights()).squeeze()
	


	movie_list = {'Animation': 1, "Children's": 1, 'Comedy': 1, 'Adventure': 2, 'Fantasy': 2, 'Romance': 3, 
				  'Drama': 4, 'Action': 2, 'Crime': 5, 'Thriller': 5, 'Horror': 5, 'Sci-Fi': 2,
				  'Documentary': 7, 'War': 10, 'Musical': 4, 'Mystery': 8, 'Film-Noir': 8, 'Western': 9}


	f = open('movies.csv','r',encoding='latin-1')
	n_row = 0
	tag_dict = {}
	y = []
	movie = np.ones((3883,5))
	index = 0
	for line in f:
		if n_row>0:
			line = line.strip('\n')
			x = line.split('::')
			tags = x[2].split('|')
			index = np.random.randint(0,len(tags))
			movie[n_row-1] = movie_emb[int(x[0])]
			y.append(movie_list[tags[index]])
		n_row+=1
	f.close() 
	y = np.array(y)
	print (y.shape)
	draw(movie,y)
	

if __name__=='__main__':
	main()