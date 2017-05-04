import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from keras.models import model_from_json
#from utils import *
import numpy as np
import pandas as pd
import csv

def main():                                                                  
    #emotion_classifier = load_model(model_2.h5)
    json_file = open('model_2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('model_2.h5')
    layer_dict = dict([layer.name, layer] for layer in loaded_model.layers[1:])

    input_img = loaded_model.input
    name_ls = ['block2_conv1']																	# jimmy : set your specific layer
    collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]


    #===== jimmy : read one picture , please your own code!!!!====#									# jimmy : read picture
    f = open('train.csv','r')

    image = []

    n_row= 0
    for row in csv.reader(f):
        if(n_row==282):
            image.append(row[1].split())
        n_row = n_row+1
    f.close()

    a = np.array(image)
    a = a.astype('float32')
    a = a/255
    a = a.reshape(1,48,48,1)
	#==============================================================


    for cnt, fn in enumerate(collect_layers):
        im = fn([a, 0]) #get the output of that layer
        fig = plt.figure(figsize=(20, 8))
        nb_filter = im[0].shape[3]
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/5, 5, i+1)                                               # jimmy : change your arrangement
            ax.imshow(im[0][0, :, :, i], cmap='Greys')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        fig.suptitle('Output of layer{} (Given image{})'.format(cnt, 546))
        #img_path = os.path.join(vis_dir, store_path)
        #if not os.path.isdir(img_path):
        #    os.mkdir(img_path)
        #fig.savefig(os.path.join(img_path,'layer{}'.format(cnt)))
        fig.savefig("sad_figure-output")                                                                # jimmy : change saved picture name

if __name__ == "__main__":
    main()

