import pickle

import data_reader
import test_Read
import numpy as np



from keras.utils import to_categorical

# validation data
test_data_file = 'input/dataset/test_stage_2.tsv'
data_test = test_Read.load_test_data(test_data_file)
test_gene_texts = data_test[:,1]
ids = data_test[:,2]
# sequentializing validation data

word_index,test_docs = data_reader.tokenizer(test_gene_texts,20000)

# x_val = data_reader.text_to_sequences(valid_gene_texts,word_index,MAX_SEQUENCE_LENGTH)
# y_val = to_categorical(np.asarray(valid_labels)) # one-hot encoding

pickle.dump([ids,test_docs],open('test2_id2doc.pkl', 'wb'))


