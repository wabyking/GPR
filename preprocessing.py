# -*- coding: utf-8 -*-
import pickle
import test_Read
import data_reader
import re
import numpy as np

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()


class Process(object):
    def __init__(self,opt,train_data_file = 'input/dataset/gap-development.tsv',valid_data_file = 'input/dataset/gap-validation.tsv',test_data_file="input/dataset/gap-test.tsv"):
        self.train_data_file = train_data_file
        self.valid_data_file = valid_data_file
        self.test_data_file = test_data_file
        self.opt=opt

        
    

    def getData(self):
    
        tmp = pickle.load(open('4kdoc2label_tokens.pkl','rb'))
        word_index, docs, labels = tmp[0], tmp[1], tmp[2]
        self.opt.word_index = word_index
        print('word_index:',len(word_index))
        print('docs:',len(docs))

        # train data loading
        x_train = data_reader.docs_to_sequences_suffix(docs,word_index,self.opt.max_sequence_length )
        y_train = labels # one-hot label encoding
        
        print('[train] Shape of data tensor:', x_train.shape)
        print('[train] Shape of label tensor:', y_train.shape)

        # validation data
        valid_temp = pickle.load(open('val_docs2label.pkl','rb'))
        val_docs, val_labels = valid_temp[0],valid_temp[1],
        # sequentializing validation data
        x_val = data_reader.docs_to_sequences_suffix(val_docs,word_index,self.opt.max_sequence_length)
        y_val = val_labels # one-hot encoding
        
        print('[Val] Shape of data tensor:', x_val.shape)
        print('[Val] Shape of label tensor:', y_val.shape)

        # word embedding lodading
        embeddings_index = data_reader.get_embedding_dict()
        print('Total %s word vectors.' % len(embeddings_index))

        # initial: random initial (not zero initial)
        embedding_matrix = np.random.random((len(word_index) + 1,self.opt.embedding_dim  ))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        self.opt.embedding_matrix = embedding_matrix
        return (x_train,y_train), (x_val,y_val)