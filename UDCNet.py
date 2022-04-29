import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
import os
import random
import logging
import utils
from sklearn.cluster import KMeans
from tqdm import tqdm
import copy
import argparse
import json
from dataload import TrainDataLoader
from trainer import Trainer

"""
if plot and batch_id < 3:
    fig = plt.figure()
    ax = plt.subplot(111)
    x = output_feats[i][:batch_con_length].detach().cpu().numpy()
    x_embedded = TSNE(n_components=2).fit_transform(x)
    plt.scatter(x_embedded[:,0], x_embedded[:,1], c=labels[i])
    fig.savefig('./EMNLP/Plot/epoch{}_dialogue_{}_{}.png'.format(epoch, batch_id, i))
    plt.close(fig)"""


class RelationModel(nn.Module):
    def __init__(self, args, bert_model, hidden_size=768, n_layers=1, bidirectional=False, dropout=0):
        super(RelationModel, self).__init__()

        self.args = args
        self.bert = bert_model
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout)

        self.dense = nn.Linear(hidden_size, hidden_size)

        self.linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        # self.bn = nn.BatchNorm1d(hidden_size, affine=False)

        self.freeze_parameters(self.bert)

    @staticmethod
    def freeze_parameters(model):
        for name, param in model.named_parameters():
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    @staticmethod
    def attention_score(input_feats, t=0.5):
        # input_feat: N*C*768
        new_feats = input_feats.transpose(1, 2)
        matrix = torch.exp(torch.matmul(input_feats, new_feats) * t)
        new_matrix = torch.zeros_like(matrix).cuda()
        for i in range(input_feats.size(0)):
            for j in range(input_feats.size(1)):
                new_matrix[i][j] = matrix[i][j] / torch.sum(matrix[i][j])
        return new_matrix

    def forward(self, inputs, mask):
        if self.args.mean_pooling:
            encoded_layer_12 = self.bert(**inputs, output_hidden_states=True, return_dict=True).last_hidden_state
            embeddings = self.dense(encoded_layer_12.mean(dim=1))
        else:
            # pooled_output = self.bert(**inputs, return_dict=True).pooler_output
            embeddings = self.bert(**inputs, return_dict=True).last_hidden_state[0]

        conversation_feats = embeddings.view(mask.size(0), -1, embeddings.size(-1))
        if self.args.lstm:
            conversation_feats, _ = self.lstm(conversation_feats)

        linear_feats = self.linear(conversation_feats)

        if self.args.ln:
            # linear_feats = linear_feats*mask.unsqueeze(2)
            # norm_feats = F.normalize(linear_feats, p=2, dim=2)
            return linear_feats
        else:
            return conversation_feats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_labels', type=int, default=4)
    parser.add_argument('--temp', type=float, default=0.1)
    parser.add_argument('--margin', type=float, default=2.0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--crop_num', type=int, default=60)
    parser.add_argument('--samples_num', type=int, default=30000)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--lstm', action='store_true')
    parser.add_argument('--mean_pooling', action='store_true')
    parser.add_argument('--ln', action='store_true')
    # parser.add_argument('--use_labels', action='store_true')
    parser.add_argument('--bert_type', type=str, default='bert', help="bert, simcse")

    parser.add_argument('--dataset', type=str, default='dialogue')
    parser.add_argument('--save_model_path', type=str, default="./EMNLP/Model")

    parser.add_argument('--device', type=str, default='0')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    SEED = random.randint(1, 100)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    savename = "UDC-Net_"
    savename += str(args.temp)
    if args.dataset == "dialogue":
        all_utterances, labels = utils.read_data(data_path="../dataset/entangled_train.json", mode='train')
        dev_utterances, dev_labels = utils.read_data(data_path="../dataset/entangled_dev.json", mode='dev')
    elif args.dataset == "irc":
        savename += "_IRC"
        args.num_labels = 14
        all_utterances, labels = utils.read_data(data_path="../dataset/irc_dialogue/irc50_train_delete.json", mode='train')
        dev_utterances, dev_labels = utils.read_data(data_path="./dataset/irc_dialogue/irc50_dev_delete.json", mode='dev')
    else:
        raise ValueError("Dataset type must be dialogue or irc.")

    if args.bert_type == "simcse":
        savename += "_SimCSE"
        args.tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
        bert_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    elif args.bert_type == "bert":
        savename += "_Bert"
        args.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        bert_model = AutoModel.from_pretrained("bert-base-uncased")
    else:
        raise ValueError("Input bert type must be bert or simcse.")

    train_loader = TrainDataLoader(args, all_utterances, labels)
    dev_loader = TrainDataLoader(args, dev_utterances, dev_labels, name="dev")
    savename += ("_Margin" + str(args.margin))
    savename += str(args.num_labels)
    savename += ("_" + str(args.samples_num))
    args.savename = savename
    """
    if args.ln:
        savename += "_LN"
    if args.lstm:
        savename += "_LSTM"
    if args.mean_pooling:
        savename += "_Pool"
    """
    args.ln = True
    args.lstm = True
    args.mean_pooling = True
    logger_name = os.path.join("./EMNLP/ConLog", "{}.txt".format(savename))
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, filename=logger_name, filemode='w')
    logger = logging.getLogger()

    log_head = "Learning Rate: {}; Random Seed: {}; Clustering Number: {}; ".format(args.learning_rate, SEED,
                                                                                    args.num_labels)
    logger.info(log_head)
    model_name = os.path.join(args.save_model_path, "model_{}".format(args.savename))
    if not os.path.exists(model_name):
        os.makedirs(model_name)

    model = RelationModel(args, bert_model, hidden_size=768).cuda()
    trainer = Trainer(args, model, logger, all_utterances)

    trainer.train(train_loader, dev_loader)
