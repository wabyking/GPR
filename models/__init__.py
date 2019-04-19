# -*- coding: utf-8 -*-

from models.LSTM1 import LSTM1
from models.CNN import CNN
from models.BiLSTM import BiLSTM
from models.TwoLayerLSTM import TwoLayerLSTM



def setup(opt):
    if opt.model == "lstm":
        model = LSTM1(opt)
    elif opt.model == "cnn":
        model = CNN(opt)
    elif opt.model == "bilstm":
        model = BiLSTM(opt)
    elif opt.model == "bilstm2":
        model = TwoLayerLSTM(opt)
    else:
        raise Exception("model not supported: {}".format(opt.model))
    return model