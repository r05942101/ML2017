from sys import argv
import csv 
import numpy as np
import numpy.random as npr
import random
def main():
	train_set = []
	f = open(argv[1],'r')
	i = 0

	for row in csv.reader(f):
		if i%18==10:
			for x in range(24):
				train_set.append(float(row[x+3]))
		
		i = i+1
	f.close()


	w = np.ones((1,9))/10
	#w = np.array([0.1,-0.1,0.1,0.1,0.1,0.1,0.2,0.2,0.6])
	w2 = 0.01
	b = 5
	lr = 10
	sigma_w = np.zeros((1,9))  #sigma_w = [[0,0,0...,0]]
	sigma_w2 = 0
	sigma_b = 0

	delta_w = np.zeros((1,9))  #delta_w = [[0,0,....,0,0]]
	delta_w2 = 0
	delta_b = 0
	zeros = np.zeros((1,9))

#lamda = 1


	for i in range(10000):
		Loss = 0
		delta_w = delta_w*zeros
		delta_b = 0
		delta_w2 = 0
		for j in range(0,len(train_set)-9,480):
			for m in range(471): 
				p_matrix = np.array([train_set[j+m:j+m+9]])
				L = train_set[j+m+9]-(b + np.sum(w*p_matrix)+ w2*(p_matrix[0][8])**2)
				Loss = Loss+L**2
			
				for k in range(9):
					delta_w[0][k] = delta_w[0][k]+(-2) * L * (p_matrix[0][k])
				delta_w2 = delta_w2 + (-2)*L*(p_matrix[0][8])**2 
				delta_b =  delta_b+(-2) * L
		for k in range(9):
			sigma_w[0][k] = (sigma_w[0][k]**2 + delta_w[0][k]**2)**0.5
			w[0][k] = w[0][k] - lr*delta_w[0][k]/sigma_w[0][k]	
		sigma_w2 = (sigma_w2**2 + delta_w2**2)**0.5
		w2 = w2 - lr*delta_w2/sigma_w2			
		sigma_b = (sigma_b**2 + delta_b**2)**0.5
		b = b - lr*delta_b/sigma_b
		#print Loss		

			
"""
#calculate RMSE
	rmse = 0
	count = 0
	temp = 0
	for i in range(0,5751,480):
		for j in range(471):
			a = np.array([train_set[i+j:i+j+9]])
			temp = b + np.sum(w*a) + w2*(a[0][8]**2)
			rmse = rmse + (train_set[i+j+9]-temp)**2
			count = count+1
	print ('learning rate = ',lr)
	rmse = (rmse/count)**0.5
	print ('RMSE = ',rmse)
"""

	f = open(argv[2],'r')
	i = 0

	test = []
	for row in csv.reader(f):
		if i%18==9:
			for j in range(9):
				test.append(int(row[j+2]))
		i = i+1
	f.close()


	ans = []
	for i in range(240):
		input_array = np.array([test[i*9:i*9+9]])
		y = b + np.sum(w*input_array) + w2*(input_array[0][8]**2)
		ans.append(y)

"""
	print ('b = ',b) 
	print 'w = '
	print w
	print ('w2 = ',w2)
	print ('sigma_w = ', sigma_w)
	print ('sigma_w2 = ', sigma_w2)
	print ('sigma_b = ',sigma_b)
"""

	f = open(argv[3],'w')
	write = csv.writer(f)
	text = ['id','value']
	write.writerow(text)
	for i in range(240):
		write.writerow(['id_'+str(i),ans[i]])
	f.close()
if __name__ == "__main__":
    main()




