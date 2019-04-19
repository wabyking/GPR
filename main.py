# -*- coding: utf-8 -*-
from  Params import Params
import argparse
from preprocessing import Process
import models

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
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score,log_loss


 


def emsemble(vals,dir_name="saved_model"):
    predicts= [] 
   
    
    for filename in os.listdir( dir_name ):
        contatenate_flag= int(filename.split("_")[-2][-1])
        val = vals[contatenate_flag]
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
        "model": ["lstm", "bilstm", "bilstm2"],
        "batch_size":[16,32,64],
        "validation_split":[0.05,0.1,0.15,0.2],
        "contatenate":[0,1],
        "lr":[0.001,0.1,0.01]       
    }
    grid_parameters ={
        "cell_type":["gru"], 
        "hidden_unit_num":[20,50,75],
        "dropout_rate" : [0.3],#,0.5,0.75,0.8,1]    ,
        "model": ["bilstm2"],
        "contatenate":[0,1],
        "lr":[0.001,0.1,0.01,0.001],
        "batch_size":[16,32,64],
        "validation_split":[0.05,0.1,0.15,0.2],
    }
    process = Process(params)
    train_uncontatenated = process.get_train()
#    val_uncontatenated = process.get_test()
    train_contatenated = process.get_train(contatenate =1)
#    val_contatenated = process.get_test(contatenate =1)
    parameters= [arg for index,arg in enumerate(itertools.product(*grid_parameters.values())) if index%args.gpu_num==args.gpu]
    for parameter in parameters:
        print(parameter)

        train = train_contatenated   if params.contatenate else train_uncontatenated
        params.setup(zip(grid_parameters.keys(),parameter))
                  
        model = models.setup(params)
        model.train(train)
        

def draw_result(predicted, val):
    print(log_loss(val,predicted)) 
    
    
    ground_label = np.array(val).argmax(axis=1)
    predicted_label = np.array(predicted).argmax(axis=1)
    print(f1_score(predicted_label ,ground_label,average='macro'))
    print(accuracy_score(predicted_label ,ground_label))
    print(confusion_matrix(predicted_label ,ground_label))
   

def test_model():
    process = Process(params)
    _ = process.get_train()  #waby: in orde to get get the test set properly [build word index parameter], actually.
    val_uncontatenated = process.get_test()
#    train_contatenated = process.get_train(contatenate =1)
    val_contatenated = process.get_test(contatenate =1)
    predicted = emsemble([val_uncontatenated,val_contatenated])    
    

    draw_result(predicted,val_contatenated[1])
    
#    predicted = emsemble(train)  
#    print(log_loss(train[1],predicted))
#    draw_result(predicted,train)

if __name__ == '__main__':
   
#    test_model()
#    train_model()
    test_model()
    
    