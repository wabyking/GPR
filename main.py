# -*- coding: utf-8 -*-
from  Params import Params
import argparse
from preprocessing import Process
import models
from sklearn.metrics import log_loss
from keras.models import load_model
import os
import numpy as np
import itertools

params = Params()
parser = argparse.ArgumentParser(description='Running Gap.')
parser.add_argument('-config', action = 'store', dest = 'config', help = 'please enter the config path.',default='config/config.ini')
parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=1)
parser.add_argument('-gpu', action = 'store', dest = 'gpu', help = 'please enter the specific gpu no.',default=0)
args = parser.parse_args()
params.parse_config(args.config)



process = Process(params)
train, val = process.getData()   


def emsemble(val,dir_name="saved_model"):
    predicts= [] 
    for filename in os.listdir( dir_name ):
        model_file = os.path.join(dir_name,filename)
        model = load_model(model_file)
        predicted = model.predict(val[0])
        print(filename + ": " + str(log_loss(val[1], predicted)))
        predicts.append(predicted)
    return np.mean(predicts,axis=0)
        
def train_model():
    grid_parameters ={
        "cell_type":["lstm","gru","rnn"], 
        "hidden_unit_num":[20,50,75,100,200],
        "dropout_rate" : [0.1,0.2,0.3],#,0.5,0.75,0.8,1]    ,
        "ablation" : [1],
        "model": ["lstm", "bilstm", "bilstm2"]
    }
    parameters= [arg for index,arg in enumerate(itertools.product(*grid_parameters.values())) if index%args.gpu_num==args.gpu]
    for parameter in parameters:
        print(parameter)
        params.setup(zip(grid_parameters.keys(),parameter))
        model = models.setup(params)
        model.train(train)
        

def draw_result(predicted, val):
    from sklearn.metrics import f1_score,confusion_matrix,accuracy_score
    print(f1_score(np.array(predicted).argmax(axis=1) ,np.array(val[1]).argmax(axis=1),average='macro'))
    
    print(confusion_matrix(np.array(predicted).argmax(axis=1) ,np.array(val[1]).argmax(axis=1)))
    print(accuracy_score(np.array(predicted).argmax(axis=1) ,np.array(val[1]).argmax(axis=1)))

def test_model():
    predicted = emsemble(val)    
    print(log_loss(val[1],predicted))
    draw_result(predicted,val)
    
#    predicted = emsemble(train)  
#    print(log_loss(train[1],predicted))
#    draw_result(predicted,train)

if __name__ == '__main__':
   
#    test_model()
#    train_model()
    test_model()
    
    