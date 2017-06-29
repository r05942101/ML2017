import pandas as pd
import xgboost as xgb
import numpy as np
#gbm = xgb.Booster({'nthread':4})
#gbm.load_model("xgboost.model")
#gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)
train_data=pd.read_csv('../train_matrix.csv')
train_label = pd.read_csv('../train_label_matrix.csv')
testing_data = pd.read_csv('../test_matrix.csv')
testing_size = testing_data.shape[0]
print (testing_size)
testing_data=xgb.DMatrix(testing_data)
print (train_data.shape)
print (train_label.shape)
train_data=xgb.DMatrix(data=train_data,label = train_label )


param = {}
param['objective'] = 'multi:softmax'
param['eta'] = 0.1
param['max_depth'] = 20
param['silent'] = 0
param['nthread'] = 4
param['num_class'] = 4
param['booster'] = 'gbtree'
param['eval_metric'] = 'merror'
param['colsample_bytree'] = 0.4


run_iter = 13
#Y_total = np.zeros((run_iter,testing_data.shape[0],3))
Y_total = np.zeros((testing_size,3))
for i in range(run_iter):

# gbm=xgb.train(train_data, objective = "multi:softmax", booster = "gbtree",
#                  eval_metric = "merror", num_round =70, 
#                  num_class = 4,eta = .2, max_depth = 14, colsample_bytree = .4)
	run_number = int(np.random.uniform(50, 80))
	gbm = xgb.train(param, train_data, run_number)
	print ("============================training end")
	name = "best"+str(i)+".model"
	gbm.save_model(name)
	pred = gbm.predict(testing_data)

	print (pred)
	for k in range(1,4):
		index = np.where( pred == k )
		print (index[0].shape)
		Y_total[index,k-1] += 1



print (Y_total)

np.savetxt('../test_out.csv', Y_total, delimiter=',')

raw_data = pd.read_csv('../testset_value.csv')
test_id = np.array(raw_data['id'][:])


# gbm = xgb.Booster({'nthread':4})
# gbm.load_model("xgboost.model")
#print(type(testing_data))


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

f = open("../result4.csv", "w")
f.write("id,status_group\n")
for i in range(pred.shape[0]):
    f.write("%s,%s\n" % (test_id[i], output[i]))
f.close()