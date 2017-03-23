from sys import argv
import csv 
import numpy as np
import numpy.random as npr
import random

def main():
	weight = [2.323721762, -0.02396138,-0.03030493,0.18270164,-0.18513731,
	-0.07276256,0.49060316,-0.51847435,0.75029463,0.28256371,0.770361413,0.00056358] # weight = [ b, w1,w2 ......w11]
	

	f = open(argv[2],'r')

	test = []
	n_row = 0
	for row in csv.reader(f):
		if n_row%18==9:
			for x in range(2,11):
				test.append(float(row[x]))
		n_row = n_row+1
	f.close()

	answer = []
	input_array = []
	for i in range(240):
		input_array = np.array([test[i*9:i*9+9]])
		w = np.array([weight[1:10]])
		y = weight[0] + np.sum(w*input_array) + weight[10]*(input_array[0][8]-input_array[0][7]) + weight[11]*input_array[0][8]**2
		answer.append(y)
	f = open(argv[3],'w')
	w = csv.writer(f)
	text = ['id','value']
	w.writerow(text)
	for i in range(240):
		w.writerow(['id_'+str(i),answer[i]])
	f.close()
if __name__ == "__main__":
    main()






