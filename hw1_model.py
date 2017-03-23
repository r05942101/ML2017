from sys import argv
import csv 
import numpy as np
import numpy.random as npr
import random

def main():
	weight = [2.0429357053105419,-0.036182,-0.01929869,0.21070292,-0.23875018
	,-0.04534373,0.51925473,-0.57038056,0.00815905,1.06501053,0.00036466211103281096]
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
		y = weight[0] + np.sum(w*input_array) + weight[10]*input_array[0][8]**2
		answer.append(y)

	f = open(argv[3],'w')
	write = csv.writer(f)
	text = ['id','value']
	write.writerow(text)
	for i in range(240):
		write.writerow(['id_'+str(i),answer[i]])
	f.close()

if __name__ == "__main__":
    main()






