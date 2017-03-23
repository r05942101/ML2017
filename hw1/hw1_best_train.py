from sys import argv
import csv 
import numpy as np
import numpy.random as npr
import random
def main():
	data = []
	for i in range(12):
		data.append([])


	f = open('train.csv','r')
	n_row = 0


	for row in csv.reader(f):
		if n_row!=0:
			if n_row%18==10:
				for x in range(3,27):
					data[(n_row-1)/360].append(float(row[x]))
		n_row = n_row+1	
	
	f.close()



#weight = np.ones((1,9))/10
	weight = np.array([[0.1,0.1,0.1,0.1,0.1,0.2,0.2,0.2,0.4]])
	weight10 = 0.1
	weight11 = 0.01 

	delta_weight = np.zeros((1,9))
	sigma_weight = np.zeros((1,9))
	delta_weight10 = 0
	sigma_weight10 = 0
	delta_weight11 = 0
	sigma_weight11 = 0

	b = 5
	delta_b = 0
	sigma_b = 0

	learning_rate = 10

	weight_sum = 0
	resetto0_array = np.zeros((1,9))
	L = 0
	LOSS = 0
	temp = 0

	for i in range(11000):  # iteration
		for j in range(12):  # 12 month
			for k in range(471):
			
				training_data = np.array([data[j][k:k+9]]) #pm2.5
			
			
				L = data[j][k+9] - (b + np.sum(training_data*weight) + weight10*(training_data[0][8]-training_data[0][7]) + weight11*training_data[0][8]**2)
			
		
				LOSS = LOSS+L**2

				for l in range(9):
					delta_weight[0][l] = delta_weight[0][l]+(-2) * L * training_data[0][l] 
				delta_weight10 = delta_weight10 + (-2) * L * (training_data[0][8]-training_data[0][7])
				delta_weight11 = delta_weight11 + (-2) * L * (training_data[0][8]**2)
				delta_b =  delta_b+(-2) * L
			
			# update
		for l in range(9):
			sigma_weight[0][l] = (sigma_weight[0][l]**2 + delta_weight[0][l]**2)**0.5
			weight[0][l] = weight[0][l] - learning_rate*delta_weight[0][l]/sigma_weight[0][l]
		sigma_weight10 = (sigma_weight10**2 + delta_weight10**2)**0.5
		if delta_weight10!=0:
			weight10 = weight10 - learning_rate*delta_weight10/sigma_weight10
		else:
			weight10 = weight10

		sigma_weight11 = (sigma_weight11**2 + delta_weight11**2)**0.5
		weight11 = weight11 - learning_rate*delta_weight11/sigma_weight11


		sigma_b = (sigma_b**2 + delta_b**2)**0.5				
		b = b - learning_rate*delta_b/sigma_b
				
	#print LOSS
	# reset
		LOSS = 0
		delta_b = 0
		delta_weight = delta_weight*resetto0_array
		delta_weight10 = 0
		delta_weight11 = 0
		

		random.shuffle(data)
			


	f = open('test_X.csv','r')


	test = []

	n_row = 0

	for row in csv.reader(f):
		if n_row%18==9:
			for x in range(2,11):
				test.append(float(row[x]))
	#elif n_row%18==9:
	#	for x in range(2,11):
	#		test[n_row/18][(n_row+1)%9].append(float(row[x]))
		n_row = n_row+1
	f.close()

	answer = []

	for i in range(240):
		temp = 0
	#for j in range(2):
	##	temp = temp + np.sum(weight[j]*input_array)
		input_array = np.array([test[i*9:i*9+9]])
		y = b + np.sum(weight*input_array) + weight10*(input_array[0][8] - input_array[0][7]) + weight11*input_array[0][8]**2
		answer.append(y)
	

	f = open('res.csv','w')
	w = csv.writer(f)
	text = ['id','value']
	w.writerow(text)
	for i in range(240):
		w.writerow(['id_'+str(i),answer[i]])
	f.close()
"""
print ('learning rate = ', learning_rate)
print ('b = ',b) 
print 'weight = '
print weight
print ('weight10 = ',weight10)
print ('weight11 = ',weight11)
print ('sigma_weight = ', sigma_weight)
print ('sigma_weight10 = ',sigma_weight10)
print ('sigma_weight11 = ',sigma_weight11)
print ('sigma_b = ',sigma_b)
"""
if __name__ == "__main__":
    main()

