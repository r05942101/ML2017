import sys 
import os
import numpy as np
import csv
import math
from sys import argv

def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1 / (1 + math.exp(gamma))
    return 1 / (1 + math.exp(-gamma))

def z(x,w,b):
	return w.dot(x.T)+b

def main():
# read feature
	epoch = 32561

	train = []
	for i in range(epoch):
		train.append([])

	index = []
	for i in range(0,106):
		if i!=14 and i!=52 and i!=105:
			index.append(i)

	dimension = len(index) 

	f = open(argv[3],'r')
	n_row = 0
	for row in csv.reader(f):
		if n_row!=0:
			for i in range(dimension):
				train[n_row-1].append(float(row[index[i]]))
		n_row = n_row+1
	f.close()

	f = open(argv[4],'r')

	n_row = 0
	for row in csv.reader(f):
		train[n_row].append(float(row[0]))
		n_row = n_row+1
	f.close()

	up50_count = 0.0
	below50_count = 0.0

	for i in range(epoch):
		if train[i][dimension] == 1:
			if up50_count == 0:
				up50 = np.matrix([train[i][0:dimension]])
			else:
				up50 = np.append(up50,[train[i][0:dimension]],axis=0)
			up50_count = up50_count+1
		else:
			if below50_count == 0:
				below50 = np.matrix([train[i][0:dimension]])
			else:
				below50 = np.append(below50,[train[i][0:dimension]],axis=0)
			below50_count = below50_count+1



# calculate mean
	mean_up50 = np.mean(up50,axis=0)
	mean_below50 = np.mean(below50,axis=0)


# calculate variance

	variance_up50 = np.zeros((dimension,dimension))
	variance_below50 = np.zeros((dimension,dimension))
	for i in range(int(up50_count)):
		variance_up50 = variance_up50 + (up50[i]-mean_up50).T.dot((up50[i]-mean_up50))
	for i in range(int(below50_count)):
		variance_below50 = variance_below50 + (below50[i]-mean_below50).T.dot((below50[i]-mean_below50))

	variance_up50 = variance_up50/epoch
	variance_below50 = variance_below50/epoch

	variance = variance_up50+variance_below50


#w_T = np.dot((mean_up50 - mean_below50).T,variance**-1)
	w_T = (mean_up50 - mean_below50).dot(variance**-1)
	b = ((mean_below50.dot(variance**-1)).dot(mean_below50.T) - (mean_up50.dot(variance**-1)).dot(mean_up50.T))/2 + math.log((up50_count/below50_count))

# read feature

	test = []
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
		x = np.matrix([test[i]])
		temp = z(x,w_T,b)
		if temp>0:
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








