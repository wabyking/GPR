
import os
#task: 1. get tokenized diction; 2. 
import test_Read
import stanfordnlp
import NLP
import numpy as np
import codecs

# global tool
nlp = None

GLOVE_DIR = "/home/dongsheng/data/resources/glove"

punctuation_list = [',',':',';','.','!','?','...','…','。']
# punctuation_list = ['.']

def get_embedding_dict():
	embeddings_index = {}
	f = codecs.open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),encoding="utf-8")
	for line in f:
		if line.strip()=='':
			continue
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	# customized dict
	f =  codecs.open(os.path.join(GLOVE_DIR, 'customized.100d.txt'),encoding="utf-8")  #
	for line in f:
		if line.strip()=='':
			continue
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	return embeddings_index


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()


def tokenizer(gene_texts,MAX_NB_WORDS):
	word_index = {}
	docs = []
	txt_count = 0
	index = 1
	if nlp is None:
		nlp=stanfordnlp.Pipeline(use_gpu=False)
	for text in gene_texts:
		txt_count+=1
		if txt_count%100==0:
			print('[tokenized txt]:',txt_count)
		doc = nlp(text)
		docs.append(doc)
		txt_matrix = NLP.get_text_matrix(doc)	# doc matrix (array)
		# == process this text matrix
		for i in range(len(txt_matrix)):
			sent_arr = txt_matrix[i]
			for j in range(len(sent_arr)):
				token = sent_arr[j].lower()
				# add to word_index
				if len(word_index)<MAX_NB_WORDS:
					if token in word_index.keys():
						continue
					else:
						word_index[token] = index
						index+=1
	return word_index,docs

# input is the generalized text; 
def text_to_sequences(gene_texts,word_index, MAX_SEQUENCE_LENGTH):
	sequences = []
	if nlp is None:
		nlp=stanfordnlp.Pipeline(use_gpu=False)
	# nlp = stanfordnlp.Pipeline()
	for text in gene_texts:
		doc = nlp(text)
		txt_matrix = NLP.get_text_matrix(doc)	# doc matrix (array)
		mention_pred = NLP.get_mention_predicate(doc)	# local
		global_pred = NLP.get_global_predicate(doc)	# global
		sequence = []
		
		# == process this text matrix
		for i in range(len(txt_matrix)):
			sent_arr = txt_matrix[i]
			for j in range(len(sent_arr)):
				token = sent_arr[j].lower()
				token_index = 0	
				if token in word_index.keys():
					token_index = word_index[token]
				# local encoding (predicates of mentions)
				local_encoding = 0 
				if token in ['aaac','bbbc','pppc','pppcs']:
					pred_pos = mention_pred[token]['predicate']
					pred_token = txt_matrix[pred_pos[0]][pred_pos[1]]
					if pred_token in word_index.keys():
						local_encoding = word_index[pred_token]
				# global encoding (punctuations or predicates)
				global_encoding = 0
				if token in punctuation_list:
					global_encoding = token_index
				else:
					if global_pred[i]['head']==j:
						global_encoding = token_index
				# concatenate
				concate = [token_index,local_encoding,global_encoding]
				sequence+=concate # add to the list
		if len(sequence)>MAX_SEQUENCE_LENGTH:
			sequence = sequence[:MAX_SEQUENCE_LENGTH]
		else:
			sequence = np.zeros(MAX_SEQUENCE_LENGTH-len(sequence),dtype=int).tolist()+sequence
		sequences.append(sequence)
	return np.asarray(sequences,dtype=int)


# input is the generalized text; 
def docs_to_sequences(docs,word_index, MAX_SEQUENCE_LENGTH):
	sequences = []
	for doc in docs:
		txt_matrix = NLP.get_text_matrix(doc)	# doc matrix (array)
		# txt_matrix = np.asarray(txt_matrix)
		mention_pred = NLP.get_mention_predicate(doc)	# local
		global_pred = NLP.get_global_predicate(doc)	# global
		sequence = []
		
		# == process this text matrix
		for i in range(len(txt_matrix)):
			sent_arr = txt_matrix[i]
			for j in range(len(sent_arr)):
				token = sent_arr[j].lower()
				token_index = 0	
				if token in word_index.keys():
					token_index = word_index[token]
				# local encoding
				local_encoding = 0 
				if token in ['aaac','bbbc','pppc','pppcs']:
					pred_pos = mention_pred[token]['predicate']
					pred_token = txt_matrix[pred_pos[0]][pred_pos[1]]
					if pred_token in word_index.keys():
						local_encoding = word_index[pred_token]
				# global encoding
				global_encoding = 0
				if token in punctuation_list:
					global_encoding = token_index
				else:
					if global_pred[i]['head']==j:
						global_encoding = token_index
				# concatenate
				concate = [token_index,local_encoding,global_encoding]
				sequence+=concate # add to the list
		if len(sequence)>MAX_SEQUENCE_LENGTH:
			sequence = sequence[:MAX_SEQUENCE_LENGTH]
		else:
			sequence = np.zeros(MAX_SEQUENCE_LENGTH-len(sequence),dtype=int).tolist()+sequence
		# print('seq:',sequence)
		sequences.append(sequence)
	return np.asarray(sequences,dtype=int)

# input is the generalized text; 
def docs_to_sequences_suffix(docs,word_index, MAX_SEQUENCE_LENGTH, contatenate=False):
	sequences = []
	a = 1
	for doc in docs:
		# print("Doc in docs:", a)
		a+=1
		txt_matrix = NLP.get_text_matrix(doc)	# doc matrix (array)
		# txt_matrix = np.asarray(txt_matrix)
		mention_pred = NLP.get_mention_predicate(doc)	# local
		global_pred = NLP.get_global_predicate(doc)	# global
		sequence = []

		# == process this text matrix
		attentions = []
		for i in range(len(txt_matrix)):
			sent_arr = txt_matrix[i]
			for j in range(len(sent_arr)):
				token = sent_arr[j].lower()
				token_index = 0
				if token in word_index.keys():
					token_index = word_index[token]
				if contatenate==True:
					sequence += [token_index]
				# local encoding
				pred_index = 0
				if token in ['aaac','bbbc','pppc','pppcs']:
					pred_pos = mention_pred[token]['predicate']
					pred_token = txt_matrix[pred_pos[0]][pred_pos[1]]
					if pred_token in word_index.keys():
						pred_index = word_index[pred_token]
					if pred_pos[1]>=j: # predicate occur after mention
						attentions+=[token_index,pred_index]
					else:
						attentions+=[pred_index,token_index]
				# global encoding
				if token in punctuation_list:
					attentions+= [token_index]
				else:
					if global_pred[i]['head']==j:
						attentions+=[token_index]
		# contatenate 
		sequence += attentions
		# print('seq/att:',len(sequence),len(attentions))
		# padding
		if len(sequence)>MAX_SEQUENCE_LENGTH:
			sequence = sequence[:MAX_SEQUENCE_LENGTH]
		else:
			sequence = np.zeros(MAX_SEQUENCE_LENGTH-len(sequence),dtype=int).tolist()+sequence
		# print('seq:',sequence)
		sequences.append(sequence)
	return np.asarray(sequences,dtype=int)


# input is the generalized text; 
def text_to_sequences_suffix(gene_texts,word_index, MAX_SEQUENCE_LENGTH):
	sequences = []
	for text in gene_texts:
		doc = nlp(text)
		txt_matrix = NLP.get_text_matrix(doc)	# doc matrix (array)
		mention_pred = NLP.get_mention_predicate(doc)	# local
		global_pred = NLP.get_global_predicate(doc)	# global
		sequence = []
		
		# == process this text matrix
		attentions = []
		for i in range(len(txt_matrix)):
			sent_arr = txt_matrix[i]	
			for j in range(len(sent_arr)):
				token = sent_arr[j].lower()
				token_index = 0	
				if token in word_index.keys():
					token_index = word_index[token]
				# local encoding
				pred_index = 0 
				if token in ['aaac','bbbc','pppc','pppcs']:
					pred_pos = mention_pred[token]['predicate']
					pred_token = txt_matrix[pred_pos[0]][pred_pos[1]]
					if pred_token in word_index.keys():
						pred_index = word_index[pred_token]
					if pred_pos[1]>=j: # predicate occur after mention
						attentions+=[token_index,pred_index]
					else:
						attentions+=[pred_index,token_index]
				# global encoding
				if token in punctuation_list:
					attentions+= [token_index]
				else:
					if global_pred[i]['head']==j:
						attentions+=[token_index]
		# contatenate
		sequence += attentions
		# padding
		if len(sequence)>MAX_SEQUENCE_LENGTH:
			sequence = sequence[:MAX_SEQUENCE_LENGTH]
		else:
			sequence = np.zeros(MAX_SEQUENCE_LENGTH-len(sequence),dtype=int).tolist()+sequence
		# print('seq:',sequence)
		sequences.append(sequence)
	return np.asarray(sequences,dtype=int)


# from keras.utils import to_categorical

# train_data_file = 'input/dataset/gap-development.tsv'
# test_data_file = 'input/dataset/gap-test.tsv'

# data_train = test_Read.load_train_data(train_data_file)
# print(data_train.shape)
# train_gene_texts = data_train[:,1]
# train_labels = data_train[:,2]

# data_test = test_Read.load_train_data(test_data_file)
# test_gene_texts = data_test[:,1]
# test_labels = data_test[:,2]

# gene_texts = train_gene_texts.tolist()+test_gene_texts.tolist()
# labels = train_labels.tolist()+test_labels.tolist()
# onehot_labels = to_categorical(np.asarray(labels))
# print(len(gene_texts))
# # sequentializing
# word_index,docs = tokenizer(gene_texts,20000)
# # x_train =docs_to_sequences(docs,word_index,50)
# # y_train = to_categorical(np.asarray(train_labels)) # one-hot encoding

# # store the data
# import pickle
# pickle.dump([word_index,docs,onehot_labels],open('4kdoc2label_tokens.pkl', 'wb'))



# # read the data
# tmp = pickle.load(open('4ktokens.pkl','rb'))
# word_index, docs = tmp[0], tmp[1]
# print('word_index:',len(word_index))
# print('docs:',len(docs))

# x_train =docs_to_sequences(docs,word_index,190)
# y_train = to_categorical(np.asarray(train_labels)) # one-hot encoding

# nlp = stanfordnlp.Pipeline()
# unique_tokens = {}
# # 1. fit on text and get tokenized dict

# # print(train_gene_texts[0:2])
# # valid data
# data_valid = test_Read.load_train_data(valid_data_file)
# valid_gene_texts = data_valid[:,1]
# valid_labels = data_valid[:,2]
