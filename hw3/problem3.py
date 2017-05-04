
from keras.models import load_model
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import csv
import itertools

def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.jet):
	"""
    This function prints and plots the confusion matrix.
    """
	cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black") 
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

#def main():
	#model_path = os.path.join(exp_dir,store_path,'model.h5')
	#emotion_classifier = load_model(model_path)



f = open('train.csv', 'r')
x = []
y = []
n_row = 0
for row in csv.reader(f):
	if(n_row>21709):
		y.append((row[0]))
		x.append(row[1].split())
	n_row = n_row+1
x_test = np.array(x) # x.shape = 28709*2304 (2304 = 48*48)
y_test = np.array(y) # y.shape = 48*48
x_test = x_test.astype('float32')
y_test = y_test.astype('float32')
y_test = np_utils.to_categorical(y_test, 7)
x_test = x_test/255

x_test = x_test.reshape(x_test.shape[0],48,48,1)


json_file = open('model_2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('model_2.h5')

np.set_printoptions(precision=2)
predictions = loaded_model.predict_classes(x_test)

conf_mat = confusion_matrix(np.argmax(y_test,axis=1),predictions)

plt.figure()
plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
plt.show()
