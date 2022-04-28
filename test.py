import torch
import torch.nn as nn
import utils

import numpy as np

all_utterances, all_labels, train_speakers = utils.readdata2("dataset/entangled_train.json", mode='train')

pseudo_labels = np.load("dataset/dialogue_Peslabels_train_epoch0.npy", allow_pickle=True).tolist()

print(len(all_labels))
print(len(pseudo_labels))
# print(all_labels)
# print(pseudo_labels)


assert len(all_labels) == len(pseudo_labels)
assert len(all_labels) == len(train_speakers)
length = len(all_labels)
pos_correct_num = 0
neg_correct_num = 0

pos_wrong_num = 0
neg_wrong_num = 0

for batch in range(length):
    truth_label_batch = all_labels[batch]
    pseudo_label_batch = pseudo_labels[batch]
    speakers = train_speakers[batch]
    assert len(truth_label_batch) == len(pseudo_label_batch)
    assert len(truth_label_batch) == len(speakers)
    for m in range(len(truth_label_batch)-1):
        for n in range(m+1, len(truth_label_batch)):
            if speakers[m] == speakers[n]:
                if truth_label_batch[m] == truth_label_batch[n]:
                    pos_correct_num += 1
                else:
                    neg_correct_num += 1
            else:
                distance = n - m
                if distance >= 3:
                    if (truth_label_batch[m] == truth_label_batch[n]) == (pseudo_label_batch[m] == pseudo_label_batch[n]):
                        neg_correct_num += 1
                    else:
                        neg_wrong_num += 1

print("Correct number is {}, wrong number is {}".format(pos_correct_num+neg_correct_num, pos_wrong_num+neg_wrong_num))
print("Correct pos number is {}, wrong pos number is {}".format(pos_correct_num, pos_wrong_num))

print("Accuracy is {}".format((pos_correct_num+neg_correct_num)/(pos_wrong_num+neg_wrong_num+pos_correct_num+neg_correct_num)))

