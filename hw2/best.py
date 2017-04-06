import sys 
import os
import numpy as np
import csv
import math
import random
from sys import argv

def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1 / (1 + math.exp(gamma))
    return 1 / (1 + math.exp(-gamma))
def main():
#choose index of feature
	epoch = 32561
	index = []
	for i in range(0,106):
		if i!=14 and i!=52 and i!=105:
			index.append(i)


	dimension = len(index) #dimension

	Y_train = []
	f = open(argv[4],'r')


	for row in csv.reader(f):
		Y_train.append(float(row[0]))
	f.close()


# read feature
	up50_train = []
	below50_train = []
	for i in range(7841):
		up50_train.append([])
	for i in range(24720):
		below50_train.append([])
	count = 0
	count1 =0
	f = open(argv[3],'r')
	n_row = 0
	for row in csv.reader(f):
		if n_row!=0:
			if Y_train[n_row-1]== 1:
				for i in range(dimension):
					up50_train[count].append(float(row[index[i]]))
				up50_train[count].append(float(Y_train[n_row-1]))
				count = count+1
			else:
				for i in range(dimension):
					below50_train[count1].append(float(row[index[i]]))
				below50_train[count1].append(float(Y_train[n_row-1]))
				count1=count1+1
		n_row = n_row+1
	f.close()

	w = np.array([[  2.16971049e-02],
		[  6.31159553e-07],
		[  5.07780982e-01],
 		[  6.76380937e-05],
 		[  7.98597608e-04],
 		[  2.49939919e-02],
 		[  1.64112016e+00],
 		[  7.64390252e-01],
 		[  3.72570765e-01],
 		[  1.01943781e+00],
 		[  1.53868571e+00],
 		[  4.60683360e-01],
 		[  6.35746999e-01],
 		[ -3.50349717e-01],
 		[ -1.21810298e+00],
 		[ -1.01207502e+00],
 		[ -8.19378464e-01],
 		[ -1.64692877e+00],
 		[ -1.55994172e+00],
 		[ -1.84517716e+00],
 		[ -1.51819664e+00],
 		[ -3.63794878e-01],
 		[ -3.50493599e-01],
 		[  4.56044452e-01],
 		[  2.20648351e+00],
 		[ -9.23746075e-01],
 		[  1.21564317e+00],
 		[ -1.56030869e+00],
 		[  1.74530987e+00],
 		[ -5.15234153e-01],
 		[ -7.03397955e-01],
 		[  8.52642399e-01],
 		[  2.86092710e-01],
 		[ -4.71244538e-01],
 		[ -7.76317598e-01],
 		[ -5.91990410e-01],
 		[ -5.30679445e-01],
 		[ -6.36230695e-01],
 		[ -1.79783013e+00],
 		[ -7.71329782e-01],
 		[  4.77061394e-01],
 		[ -1.53716487e+00],
 		[ -1.13165891e+00],
 		[ -1.05756789e+00],
 		[ -8.52044555e-01],
 		[ -5.56473327e-01],
 		[ -5.75563220e-02],
 		[ -8.00397496e-03],
 		[ -2.90144683e-01],
 		[ -2.14931974e-02],
 		[ -1.00068568e+00],
 		[  1.17719843e+00],
 		[ -2.15614367e-01],
 		[  4.22535225e-02],
 		[  7.62564763e-03],
 		[ -9.65398939e-02],
 		[  2.16610795e+00],
 		[ -4.90695583e-01],
 		[ -1.63710875e-01],
 		[ -2.51176450e-01],
 		[ -4.39917601e-01],
 		[ -1.14579955e-01],
 		[  1.59405454e+00],
 		[  5.71069101e-01],
 		[ -4.34633113e-01],
 		[ -5.18956543e-01],
 		[  3.59419701e-01],
 		[ -8.90403922e-02],
 		[  1.24031657e-01],
		[  3.41294302e-01],
 		[  6.86852345e-01],
 		[  1.00146166e+00],
 		[  7.07315878e-01],
 		[ -5.31187464e-01],
 		[  5.62806368e-01],
 		[  1.59675862e-01],
 		[ -1.09856329e+00],
 		[  9.49215179e-02],
 		[  8.50092752e-02],
 		[  7.85570259e-02],
 		[ -9.97008033e-02],
 		[  3.99041698e-01],
 		[  8.11285606e-01],
 		[  9.58513126e-01],
 		[  3.17893670e-01],
 		[  8.41787754e-01],
 		[ -4.23759087e-01],
 		[ -3.89037205e-02],
 		[ -2.94059635e-01],
 		[ -1.00352984e+00],
 		[ -1.63647005e-01],
 		[  6.13658217e-01],
 		[  5.91460742e-02],
 		[ -4.13612216e-02],
 		[  8.76865640e-02],
 		[  1.28568129e-01],
 		[ -4.63910502e-01],
 		[ -6.02942340e-02],
 		[ -1.37402529e-01],
 		[ -5.19158435e-02],
 		[  4.56634286e-01],
 		[ -3.49122838e-01],
 		[  8.64424297e-01],
 		[  0.0]])

	b =  -5.03105876


	learning_rate = 0.0000001
	mini_batch = 7481*2

	delta_w = np.zeros((dimension,1))
	delta_b = 0
	reset = np.zeros((dimension,1))

	sigma_w = np.zeros((dimension,1))
	sigma_b = 0


	f = np.zeros((mini_batch,1))

	for i in range(4):
		data = np.array([up50_train[0]])
		for m in range(mini_batch/2-1):
			data = np.append(data,[up50_train[m+1]],axis=0)  #data = mini_batch * dimension+1
		for m in range(mini_batch/2):
			data = np.append(data,[below50_train[m]],axis=0)
		Z = data.dot(w) + b
		
		for m in range(mini_batch):
			f[m] = sigmoid(Z[m])

		for m in range(mini_batch):
			if data[m][dimension]==1:
				for k in range(dimension):
					delta_w[k] = delta_w[k]+(1-f[m])*data[m][k]*-1
				delta_b = delta_b +(1-f[m])*(-1)
			else:	
				for k in range(dimension):
					delta_w[k] = delta_w[k]+(f[m])*data[m][k]
				delta_b = delta_b + (f[m])

		#print delta_w

		for k in range(dimension):
			sigma_w[k] = (sigma_w[k]**2+delta_w[k]**2)**0.5
			if delta_w[k] != 0:
				w[k] = w[k] - learning_rate*delta_w[k]/sigma_w[k]
			else:
				w[k] = w[k]
		sigma_b = (sigma_b**2+delta_b**2)**0.5
		if delta_b!=0:
			b = b -learning_rate*delta_b/sigma_b
		else:
			b = b
		delta_w = delta_w*reset
		delta_b = 0

		random.shuffle(up50_train)
		random.shuffle(below50_train)


	test= []
	for i in range(16281):
		test.append([])

	f = open(argv[5],'r')
	n_row = 0
	for row in csv.reader(f):
		if n_row!=0:
			for i in range(dimension):
				test[n_row-1].append(float(row[index[i]]))
		n_row = n_row+1
	f.close()

	answer = []
	for i in range(16281):
		test_data = np.array([test[i]])
		test_data = np.append(test_data,[[0]],axis=1)
		z = test_data.dot(w)+b
		if z>0:
			answer.append(int(1))
		else:
			answer.append(int(0))

	f = open(argv[6],'w')
	w = csv.writer(f)
	text = ['id','label']
	w.writerow(text)
	for i in range(16281):
		w.writerow([i+1,answer[i]])
	f.close()


if __name__ == "__main__":
    main()




