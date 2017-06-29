import pandas as pd
import xgboost as xgb
import numpy as np
import sys



def main():

	testing_data = pd.read_csv('test_matrix.csv')
	testing_size = testing_data.shape[0]
	print (testing_size)
	testing_data=xgb.DMatrix(testing_data)

	raw_data = pd.read_csv('testset_value.csv')
	test_id = np.array(raw_data['id'][:])


	Y_total = np.zeros((testing_size,3))

	for i in range(2,13):
		model_name = "best"+str(i)+".model"
		gbm = xgb.Booster({'nthread':4})
		gbm.load_model(model_name)
		pred = gbm.predict(testing_data)
		for k in range(1,4):
			index = np.where( pred == k )
			print (index[0].shape)
			Y_total[index,k-1] += 1


	output = []
	for i in range(Y_total.shape[0]):
		max_index = np.argmax(Y_total[i,:])
		if max_index == 0:
			output.append("functional")
		elif max_index == 1:
			output.append("functional needs repair")
		elif max_index == 2:
			output.append("non functional")	


	#print (output)

	f = open("result4.csv", "w")
	f.write("id,status_group\n")
	for i in range(pred.shape[0]):
	    f.write("%s,%s\n" % (test_id[i], output[i]))
	f.close()


if __name__=='__main__':
    main()


