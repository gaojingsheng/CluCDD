import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
import os
import random
import logging
import utils
from utils import calculate_purity_scores, calculate_shen_f_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from tqdm import tqdm
import copy
import argparse
import json

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE



class RelationModel(nn.Module):
    def __init__(self, bert_model, hidden_size=768, n_layers=1, uselstm=True, bidirectional=False, freeze_bert=False, dropout=0):
        super(RelationModel, self).__init__()
        
        self.bert = bert_model
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True, \
                                        bidirectional=bidirectional, dropout=dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.uselstm = uselstm
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU()
        )
        # self.bn = nn.BatchNorm1d(hidden_size, affine=False)
        
        self.freeze_parameters(self.bert)

    def freeze_parameters(self,model):
        for name, param in model.named_parameters():  
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
    
    def attention_score(self, input_feats, T=0.5):
        # input_feat: N*C*768
        new_feats = input_feats.transpose(1,2)
        matrix = torch.exp(torch.matmul(input_feats, new_feats)*T)
        new_matrix = torch.zeros_like(matrix).cuda()
        for i in range(input_feats.size(0)):
            for j in range(input_feats.size(1)):
                new_matrix[i][j] = matrix[i][j]/torch.sum(matrix[i][j])
        return new_matrix

    def forward(self, inputs, mask):
            
        # if self.args.mean_pooling:
        encoded_layer_12 = self.bert(**inputs, output_hidden_states=True, return_dict=True).last_hidden_state
        embeddings = self.dense(encoded_layer_12.mean(dim = 1))
        # else:        
        #     # pooled_output = self.bert(**inputs, return_dict=True).pooler_output
        #     embeddings = self.bert(**inputs, return_dict=True).last_hidden_state[0]

        conversation_feats = embeddings.view(mask.size(0), -1, embeddings.size(-1)) 

        conversation_feats, _ = self.lstm(conversation_feats)

        linear_feats = self.linear(conversation_feats)

        # linear_feats = linear_feats*mask.unsqueeze(2)
        norm_feats = F.normalize(linear_feats, p=2, dim=2)
        return linear_feats
        

class TrainDataLoader(object):
    def __init__(self, args, all_utterances, labels, name='train'):
        self.batch_size = args.batch_size

        self.all_utterances_batch = [all_utterances[i:i+self.batch_size] \
                                    for i in range(0, len(all_utterances), self.batch_size)]
        self.labels_batch = [labels[i:i+self.batch_size] \
                            for i in range(0, len(labels), self.batch_size)]

        print("labels_batch",len(self.labels_batch))
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
            temp = utterances[i] + [self.padding_sentence]*(max_conversation_length-conversations_length[i])
            padded_conversations.extend(temp)
        return padded_conversations, conversations_length, max_conversation_length

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= self.batch_num:
            raise IndexError

        utterances = self.all_utterances_batch[key]
        labels = self.labels_batch[key]
        padded_conversations, conversations_length, max_conversation_length = self.padding_conversation(utterances, labels)
        mask = torch.arange(max_conversation_length).expand(len(conversations_length), max_conversation_length) \
                                < torch.Tensor(conversations_length).unsqueeze(1)
        # padded_conversations: NC * sentences
        return padded_conversations, labels, mask




class Trainer(object):
    def __init__(self, args, model, logger, all_utterances, loss_fn=ContrastiveLoss()):
        self.args = args
        self.model = model
        self.logger = logger
        self.loss_fn = loss_fn
        self.utterances = all_utterances
        params = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(params, lr=args.learning_rate)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_labels = args.num_labels
        self.epoch_num = args.epoch_num

    def evaluate(self, test_loader, epoch=0, mode="dev", first_time = False, plot = False):
        predicted_labels = []
        truth_labels = []
        self.model.eval()
        with torch.no_grad():
            for batch_id, batch in enumerate(tqdm(test_loader)):
                padded_conversations, labels, mask = batch
                inputs = self.args.tokenizer(padded_conversations, padding=True, truncation=True, return_tensors="pt")
                for things in inputs:
                    inputs[things] = inputs[things][:,:self.args.crop_num].cuda()
                mask = mask.cuda()
                if first_time:
                    embeddings = self.model.bert(**inputs, return_dict=True).pooler_output
                    conversation_feats = embeddings.view(mask.size(0), -1, embeddings.size(-1)) 
                    linear_feats = conversation_feats*mask.unsqueeze(2)
                    output_feats = F.normalize(linear_feats, p=2, dim=2)
                    # sim_matrix = self.model.attention_score(norm_feats, T=0.5)
                    # output = torch.matmul(sim_matrix, norm_feats)
                    # output_feats = torch.cat((norm_feats, output), dim=-1)
                else:
                    output_feats = self.model(inputs, mask)

                for i in range(len(labels)):
                    batch_con_length = torch.sum(mask[i])
                    km = KMeans(n_clusters = self.num_labels).fit(output_feats[i][:batch_con_length].detach().cpu().numpy())
                    pseudo_labels =  km.labels_
                    predicted_labels.append(pseudo_labels)
                    truth_labels.append(labels[i])
                    """
                    if plot and batch_id < 3:
                        fig = plt.figure()
                        ax = plt.subplot(111)
                        x = output_feats[i][:batch_con_length].detach().cpu().numpy()
                        x_embedded = TSNE(n_components=2).fit_transform(x)
                        plt.scatter(x_embedded[:,0], x_embedded[:,1], c=labels[i])
                        fig.savefig('./IJCAI/Plot/epoch{}_dialogue_{}_{}.png'.format(epoch, batch_id, i))
                        plt.close(fig)"""

        purity_score = utils.compare(predicted_labels, truth_labels, 'purity')
        nmi_score = utils.compare(predicted_labels, truth_labels, 'NMI')
        ari_score = utils.compare(predicted_labels, truth_labels, 'ARI')
        shen_f_score = utils.compare(predicted_labels, truth_labels, 'shen_f')
        loc3_score = utils.compare(predicted_labels, truth_labels, 'Loc3')
        log_msg = "purity_score: {}, nmi_score: {}, ari_score: {}, shen_f_score: {}, loc3_score: {}".format(
                        round(purity_score, 4), round(nmi_score, 4), round(ari_score, 4), round(shen_f_score, 4), round(loc3_score, 4))
        self.logger.info(log_msg)
        return

    def train(self, train_loader, dev_loader):
        step_cnt = 0 
        # self.evaluate(dev_loader, first_time=True)
        # self.evaluate(dev_loader, plot=True)
        # train_loader = self.evaluate(train_loader, mode="train", first_time=True)
        self.model.train()
        for epoch in tqdm(range(self.epoch_num)):
            epoch_loss = 0
            for i, batch in enumerate(tqdm(train_loader)):
                step_cnt += 1
                padded_conversations, labels, mask = batch
                inputs = self.args.tokenizer(padded_conversations, padding=True, truncation=True, return_tensors="pt")
                for things in inputs:
                    inputs[things] = inputs[things][:,:self.args.crop_num].cuda()
                mask = mask.cuda()
                output = self.model(inputs, mask)
                loss = torch.zeros(1, dtype=torch.float32).to(self.device)
                for batch in range(len(labels)):
                    train_labels = labels[batch]
                    for m in range(len(train_labels)-1):
                        for n in range(m+1, len(train_labels)):
                            loss += self.loss_fn(output[batch][m], output[batch][n], train_labels[m]!=train_labels[n])
                loss = loss/((len(train_labels)-1)**2*self.args.batch_size)
                log_msg = "Epoch : {}, batch: {}/{}, step: {},loss: {}".format(epoch, i, len(train_loader), step_cnt, round(loss.data.item(), 4))
                self.logger.info(log_msg)
                epoch_loss += loss.data.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if (step_cnt % 100 == 0 and epoch == 0) or step_cnt % 1000 == 0:
                    self.evaluate(dev_loader, first_time=True)
                    self.evaluate(dev_loader, epoch=epoch+1, plot=False)
                    self.model.train()
            log_msg = "Epoch average loss is: {}".format(round(epoch_loss/len(train_loader), 4))
            self.logger.info(log_msg)
            model_name = os.path.join("IJCAI/SubModel", "model{}_epoch{}".format(self.args.num_labels, epoch))
            log_msg = "Saving model at '{}'".format( model_name)
            self.logger.info(log_msg)
            torch.save(self.model.state_dict(), model_name)
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_labels', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--crop_num', type=int, default=60)
    parser.add_argument('--learning_rate', type=float, default=5e-4)

    parser.add_argument('--berttype', type=str, default='bert', help="bert, simcse")
    
    parser.add_argument('--dataset', type=str, default='dialogue')

    parser.add_argument('--device', type=str, default='0')

    args = parser.parse_args() 

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    SEED = random.randint(1,100)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    savename = "DCC-Net_"

    all_utterances, labels = utils.ReadData(datapath="dataset/splited/entangled{}_train.json".format(args.num_labels),
                                            mode='train')
    dev_utterances, dev_labels = utils.ReadData(datapath="dataset/splited/entangled{}_dev.json".format(args.num_labels),
                                                mode='dev')


    if args.berttype == "simcse":
        savename += "_SimCSE"
        args.tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
        bert_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    elif args.berttype == "bert":
        savename += "_Bert"
        args.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        bert_model = AutoModel.from_pretrained("bert-base-uncased")
    else:
        raise ValueError("Input bert type must be bert or simcse.")
    
    train_loader = TrainDataLoader(args, all_utterances, labels)
    dev_loader = TrainDataLoader(args, dev_utterances, dev_labels, name="dev")
    
    savename += str(args.num_labels)

    logger_name = os.path.join("./IJCAI/SessionLog", "{}.txt".format(savename))
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, filename=logger_name, filemode='w')
    logger = logging.getLogger()
    
    log_head = "Learning Rate: {}; Random Seed: {}; Clustering Number: {}; ".format(args.learning_rate, SEED, args.num_labels)
    logger.info(log_head)

    model = RelationModel(bert_model, hidden_size=768).cuda()
    trainer = Trainer(args, model, logger, all_utterances)

    trainer.train(train_loader, dev_loader)