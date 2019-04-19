from Data import DatasetSchema
import numpy as np

# generalized to: AAAC, BBBC, PPPC/PPPCS

def main():
    print("Hello World")
    with open("input/dataset/gap-development.tsv", encoding='utf8') as f:
        content = f.readlines()
    content = [x.rstrip() for x in content]
    header = content[0]
    for line in content[1:100]:
        data = DatasetSchema(line)
        generalized_txt = data.get_generalized_text()
        label_A = data.get_A_coref()
        label_B = data.get_B_coref()
        a_Pos = data.get_A_pos()
        b_Pos = data.get_B_pos()
        p_Pos = data.get_Pronoun_pos()
        label = 0
        if label_A in ['TRUE','True','true'] and label_B in ['FALSE','False','false']:
            label = 0
        elif label_B in ['TRUE','True','true'] and label_A in ['FALSE','False','false']:
            label = 1
        else:
            label = 2
        print(generalized_txt,label)


def load_train_data(tsv_file_path):
    with open(tsv_file_path, encoding='utf8') as f:
        content = f.readlines()
    content = [x.rstrip() for x in content]
    header = content[0]
    res = []
    for line in content[1:]:
        data = DatasetSchema(line)
        orig_txt = data.get_text()
        generalized_txt = data.get_generalized_text()
        # below is to get exact sentences
        sentences = generalized_txt.split('.')
        exact_sents = []
        for sent in sentences:
            if 'AAAC' in sent or 'BBBC' in sent or 'PPPC' in sent or 'PPPCS' in sent:
                exact_sents.append(sent)
        exact_txt = '.'.join(exact_sents)
        # end of previous below
        label_A = data.get_A_coref()
        label_B = data.get_B_coref()
        a_Pos = data.get_A_pos()
        b_Pos = data.get_B_pos()
        p_Pos = data.get_Pronoun_pos()
        if label_A in ['TRUE','True','true'] and label_B in ['FALSE','False','false']:
            label = 0
        elif label_B in ['TRUE','True','true'] and label_A in ['FALSE','False','false']:
            label = 1
        else:
            label = 2
        res.append([orig_txt,exact_txt,label])
    return np.array(res)


def load_test_data(tsv_file_path):
    with open(tsv_file_path, encoding='utf8') as f:
        content = f.readlines()
    content = [x.rstrip() for x in content]
    header = content[0]
    res = []
    for line in content[1:]:
        data = DatasetSchema(line)
        orig_txt = data.get_text()
        generalized_txt = data.get_generalized_text()
        # below is to get exact sentences
        sentences = generalized_txt.split('.')
        exact_sents = []
        for sent in sentences:
            if 'AAAC' in sent or 'BBBC' in sent or 'PPPC' in sent or 'PPPCS' in sent:
                exact_sents.append(sent)
        exact_txt = '.'.join(exact_sents)
        # end of previous below
        a_Pos = data.get_A_pos()
        b_Pos = data.get_B_pos()
        p_Pos = data.get_Pronoun_pos()
        samp_id = data.get_id()
        res.append([orig_txt,exact_txt,samp_id])
    return np.array(res)

# if __name__ == '__main__':
#     main()


# data_valid = load_train_data("input/dataset/gap-test.tsv")
# valid_labels = data_valid[:,2]
# print(valid_labels.tolist().count('0')) # 187/454 => 45.15% (majority voting)
# print(valid_labels.tolist().count('1')) # 205       
# print(valid_labels.tolist().count('2')) # 62
