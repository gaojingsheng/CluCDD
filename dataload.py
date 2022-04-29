import torch
import torch.nn as nn


class TrainDataLoader(object):
    def __init__(self, args, all_utterances, labels, name='train'):
        self.batch_size = args.batch_size

        self.all_utterances_batch = [all_utterances[i:i + self.batch_size] \
                                     for i in range(0, len(all_utterances), self.batch_size)]
        self.labels_batch = [labels[i:i + self.batch_size] \
                             for i in range(0, len(labels), self.batch_size)]

        print("labels_batch", len(self.labels_batch))
        assert len(self.all_utterances_batch) == len(self.labels_batch)

        self.batch_num = len(self.all_utterances_batch)
        print("{} batches created in {} set.".format(self.batch_num, name))

        self.padding_sentence = "This is a padding sentence: bala bala bala123!"

    def __len__(self):
        return self.batch_num

    def padding_conversation(self, utterances, labels):
        conversations_length = [len(x) for x in utterances]
        max_conversation_length = max(conversations_length)
        padded_conversations = []
        for i in range(len(conversations_length)):
            for j in range(len(utterances[i])):
                if len(utterances[i][j]) > 600:
                    utterances[i][j] = utterances[i][j][:600]
            temp = utterances[i] + [self.padding_sentence] * (max_conversation_length - conversations_length[i])
            padded_conversations.extend(temp)
        return padded_conversations, conversations_length, max_conversation_length

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= self.batch_num:
            raise IndexError

        utterances = self.all_utterances_batch[key]
        labels = self.labels_batch[key]
        padded_conversations, conversations_length, max_conversation_length = self.padding_conversation(utterances,
                                                                                                        labels)
        mask = torch.arange(max_conversation_length).expand(len(conversations_length), max_conversation_length) \
               < torch.Tensor(conversations_length).unsqueeze(1)
        # padded_conversations: NC * sentences
        return padded_conversations, labels, mask
