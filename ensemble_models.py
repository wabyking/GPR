import numpy as np

def combine_model_predictions(file1, file2):
    f1 = open(file1)
    predictions1 = f1.readlines()
    f2 = open(file2)
    predictions2 = f2.readlines()
    file_new_probabilities = open('merged_results.csv', 'w')
    print("ID,A,B,NEITHER",file=file_new_probabilities)
    for line1, line2 in zip(predictions1[1:], predictions2[1:]):
        parts1 = line1.split(',')
        parts2 = line2.split(',')
        devid = parts1[0]
        probs1 = [float(x) for x in parts1[1:]]
        probs2 = [float(x) for x in parts2[1:]]
        probs_sum = np.array(probs1) + np.array(probs2)
        probs_sum /= 2
        print(devid, ",".join([str(x) for x in probs_sum]), sep=',', file=file_new_probabilities)


if __name__ == '__main__':
    combine_model_predictions("submission_lucas1.txt", "submission_lucas1.txt")
