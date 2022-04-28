import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import utils
from transformers import AutoModel, AutoTokenizer
import numpy as np
from tqdm import tqdm
import argparse
import json
import os
from utils import calculate_purity_scores, calculate_shen_f_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans, AffinityPropagation, DBSCAN
from sklearn.mixture import GaussianMixture

class RelationModel(nn.Module):
    def __init__(self, args, bert_model, hidden_size=768, n_layers=1, bidirectional=False, dropout=0):
        super(RelationModel, self).__init__()

        self.args = args
        self.bert = bert_model
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True, \
                                        bidirectional=bidirectional, dropout=dropout)

        self.dense = nn.Linear(hidden_size, hidden_size)

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


    def forward(self, inputs, mask):
        if self.args.mean_pooling:
            encoded_layer_12 = self.bert(**inputs, output_hidden_states=True, return_dict=True).last_hidden_state
            embeddings = self.dense(encoded_layer_12.mean(dim = 1))
        else:        
            # pooled_output = self.bert(**inputs, return_dict=True).pooler_output
            embeddings = self.bert(**inputs, return_dict=True).last_hidden_state[0]

        conversation_feats = embeddings.view(mask.size(0), -1, embeddings.size(-1)) 
        if self.args.uselstm:
            conversation_feats, _ = self.lstm(conversation_feats)

        linear_feats = self.linear(conversation_feats)

        if self.args.ln:
            # linear_feats = linear_feats*mask.unsqueeze(2)
            norm_feats = F.normalize(linear_feats, p=2, dim=2)
            return linear_feats
        else:
            return conversation_feats

class Trainer(object):
    def __init__(self, args, model, logger):
        self.args = args
        self.model = model
        self.logger = logger
        self.num_labels = args.num_labels

    def evaluate(self, test_loader, epoch=0, mode="dev", first_time = False):
        predicted_labels = []
        truth_labels = []
        self.model.eval()
        with torch.no_grad():
            for i,batch in tqdm(enumerate(test_loader)):
                if i == 10:
                    break
                padded_conversations, labels, mask = batch
                inputs = self.args.tokenizer(padded_conversations, padding=True, truncation=True, return_tensors="pt")
                for things in inputs:
                    inputs[things] = inputs[things][:,:self.args.crop_num].cuda()
                mask = mask.cuda()
                output_feats = self.model(inputs, mask)
                for i in range(len(labels)):
                    batch_con_length = torch.sum(mask[i])
                    if self.args.methods == "kmeans":
                        km = KMeans(n_clusters = self.num_labels).fit(output_feats[i][:batch_con_length].detach().cpu().numpy())
                        pseudo_labels =  km.labels_
                    elif self.args.methods == "ap":
                        af = AffinityPropagation().fit(output_feats[i][:batch_con_length].detach().cpu().numpy())
                        pseudo_labels =  af.labels_
                    elif self.args.methods == "dbscan":
                        db = DBSCAN(eps=1, min_samples=1).fit(output_feats[i][:batch_con_length].detach().cpu().numpy())
                        pseudo_labels =  db.labels_
                    elif self.args.methods == "gauss":
                        gm = GaussianMixture(n_components = self.num_labels).fit(output_feats[i][:batch_con_length].detach().cpu().numpy())
                        pseudo_labels =  gm.predict(output_feats[i][:batch_con_length].detach().cpu().numpy())
                    predicted_labels.append(pseudo_labels)
                    print(pseudo_labels)
                    truth_labels.append(labels[i])

        purity_score = utils.compare(predicted_labels, truth_labels, 'purity')
        nmi_score = utils.compare(predicted_labels, truth_labels, 'NMI')
        ari_score = utils.compare(predicted_labels, truth_labels, 'ARI')
        shen_f_score = utils.compare(predicted_labels, truth_labels, 'shen_f')
        loc3_score = utils.compare(predicted_labels, truth_labels, 'Loc3')
        one2one_score = utils.compare(predicted_labels, truth_labels, 'one2one')
        log_msg = "purity_score: {}, nmi_score: {}, ari_score: {}, shen_f_score: {}, loc3_score: {}, one2one_score: {}".format(
                        round(purity_score, 4), round(nmi_score, 4), round(ari_score, 4), round(shen_f_score, 4), round(loc3_score, 4), round(one2one_score, 4))
        self.logger.info(log_msg)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_labels', type=int, default=4)
    parser.add_argument('--uselstm', action='store_true')
    parser.add_argument('--mean_pooling', action='store_true')
    parser.add_argument('--ln', action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--crop_num', type=int, default=60)
    parser.add_argument('--dataset', type=str, default='dialogue')
    parser.add_argument('--methods', type=str, default='kmeans')
    parser.add_argument('--save_model_path', type=str, default="./IJCAI/Model")

    parser.add_argument('--device', type=str, default='0')

    args = parser.parse_args() 

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device


    # all_utterances, labels = utils.readdata(datapath="dataset/entangled_train.json", mode='train')
    dev_utterances, dev_labels = utils.readdata(datapath="dataset/entangled_dev.json", mode='dev')

    args.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModel.from_pretrained("bert-base-uncased")
    # train_loader = TrainDataLoader(args, all_utterances, labels)
    dev_loader = TrainDataLoader(args, dev_utterances, dev_labels, name="dev")
    # savename = "test_{}_num_{}".format(args.methods, args.num_labels)
    savename = "one2one_test"
    logger_name = os.path.join("./IJCAI/TestLog", "{}.txt".format(savename))
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, filename=logger_name, filemode='w')
    logger = logging.getLogger()

    log_head = "Clustering Method: {}; Clustering Number for Kmeans: {}; ".format(args.methods, args.num_labels)
    logger.info(log_head)
    model_name = os.path.join(args.save_model_path, "model_UDC-Net__Bert_Margin2.04_25000", "step_28000.pkl")

    model = RelationModel(args, bert_model, hidden_size=768).cuda()
    # print("here1")
    model.load_state_dict(torch.load(model_name))
    trainer = Trainer(args, model, logger)

    trainer.evaluate(dev_loader)