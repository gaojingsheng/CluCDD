import utils
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import tqdm
import torch.nn.functional as F


def cal_metrics(predicted_labels, truth_labels):

    purity_score = utils.compare(predicted_labels, truth_labels, 'purity')
    nmi_score = utils.compare(predicted_labels, truth_labels, 'NMI')
    ari_score = utils.compare(predicted_labels, truth_labels, 'ARI')
    shen_f_score = utils.compare(predicted_labels, truth_labels, 'shen_f')
    loc3_score = utils.compare(predicted_labels, truth_labels, 'Loc3')
    return purity_score, nmi_score, ari_score, shen_f_score, loc3_score


def norm_embedding(embedding):
    """
    Normalization before cos similarity clustering
    """

    eps = 1e-6
    nm = np.sqrt((repre ** 2).sum(axis=1))[:, None]
    return embedding / (nm + eps)


class Trainer(object):
    def __init__(self, args, model, logger, all_utterances):
        self.args = args
        self.model = model
        self.logger = logger
        self.loss_fn = ContrastiveLoss(margin=self.args.margin)
        self.info_loss = InfoNCELoss(temp=self.args.temp)

        self.utterances = all_utterances
        params = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(params, lr=args.learning_rate)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_labels = args.num_labels
        self.epoch_num = args.epoch_num

    def evaluate(self, test_loader, epoch=0, mode="dev", first_time=False):
        predicted_labels = []
        truth_labels = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                padded_conversations, labels, mask = batch
                inputs = self.args.tokenizer(padded_conversations, padding=True, truncation=True, return_tensors="pt")
                for things in inputs:
                    inputs[things] = inputs[things][:, :self.args.crop_num].cuda()
                mask = mask.cuda()
                if first_time:
                    embeddings = self.model.bert(**inputs, return_dict=True).pooler_output
                    conversation_feats = embeddings.view(mask.size(0), -1, embeddings.size(-1))
                    linear_feats = conversation_feats * mask.unsqueeze(2)
                    output_feats = F.normalize(linear_feats, p=2, dim=2)
                    # sim_matrix = self.model.attention_score(norm_feats, T=0.5)
                    # output = torch.matmul(sim_matrix, norm_feats)
                    # output_feats = torch.cat((norm_feats, output), dim=-1)
                else:
                    output_feats = self.model(inputs, mask)

                for i in range(len(labels)):
                    batch_con_length = torch.sum(mask[i])
                    clustering_array = output_feats[i][:batch_con_length].detach().cpu().numpy()
                    norm_clustering_array = norm_embedding(clustering_array)
                    # print(clustering_array)
                    km = KMeans(n_clusters=self.num_labels).fit(norm_clustering_array)
                    pseudo_labels = km.labels_
                    predicted_labels.append(pseudo_labels)
                    truth_labels.append(labels[i])
        purity_score, nmi_score, ari_score, shen_f_score, loc3_score = cal_metrics(predicted_labels,truth_labels)
        log_msg = "purity_score: {}, nmi_score: {}, ari_score: {}, shen_f_score: {}, loc3_score: {}".format(
            round(purity_score, 4), round(nmi_score, 4), round(ari_score, 4), round(shen_f_score, 4),
            round(loc3_score, 4))
        self.logger.info(log_msg)
        return

    def train(self, train_loader, dev_loader):
        writer = SummaryWriter('runs/' + self.args.savename)
        step_cnt = 0
        # self.evaluate(dev_loader, first_time=True)
        # self.evaluate(dev_loader)
        # train_loader = self.evaluate(train_loader, mode="train", first_time=True)
        self.model.train()
        for epoch in tqdm(range(self.epoch_num)):
            epoch_loss = 0
            for i, content in enumerate(tqdm(train_loader)):
                if i * self.args.batch_size > self.args.samples_num:
                    break

                step_cnt += 1
                padded_conversations, labels, mask = content
                inputs = self.args.tokenizer(padded_conversations, padding=True, truncation=True, return_tensors="pt")
                for things in inputs:
                    inputs[things] = inputs[things][:, :self.args.crop_num].cuda()
                mask = mask.cuda()
                output = self.model(inputs, mask)
                """
                loss = torch.zeros(1, dtype=torch.float32).to(self.device)
                for batch in range(len(labels)):
                    train_labels = labels[batch]
                    for m in range(len(train_labels)-1):
                        for n in range(m+1, len(train_labels)):
                            loss += self.loss_fn(output[batch][m], output[batch][n], train_labels[m]!=train_labels[n])
                loss = loss/((len(train_labels)-1)**2*self.args.batch_size)
                """
                loss = torch.zeros(1, dtype=torch.float32).to(self.device)
                loss_num = 0
                for batch in range(len(labels)):
                    train_labels = labels[batch]
                    for m in range(len(train_labels) - 1):
                        for n in range(m + 1, len(train_labels)):
                            if train_labels[m] == train_labels[n]:
                                loss += self.info_loss(output[batch][m], output[batch][n],
                                                       output[batch][torch.arange(output.size(1)) != m])
                                loss_num += 1
                loss = loss / loss_num
                writer.add_scalar('loss', loss.data.item(), global_step=step_cnt)
                log_msg = "Epoch : {}, batch: {}/{}, step: {},loss: {}".format(epoch, i, len(train_loader), step_cnt,
                                                                               round(loss.data.item(), 4))
                self.logger.info(log_msg)
                epoch_loss += loss.data.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.args.dataset == "dialogue":
                    if step_cnt % 10 == 0 and epoch == 0:
                        # self.evaluate(dev_loader, first_time=True)
                        self.evaluate(dev_loader)
                        self.model.train()
                    elif step_cnt % 1000 == 0:
                        self.evaluate(dev_loader)
                        self.model.train()
                        model_name = os.path.join(self.args.save_model_path, "model_{}".format(self.args.savename),
                                                  "step_{}.pkl".format(step_cnt))
                        torch.save(self.model.state_dict(), model_name)
                elif self.args.dataset == "irc":
                    if step_cnt % 30 == 0 and epoch == 0:
                        # self.evaluate(dev_loader, first_time=True)
                        self.evaluate(dev_loader)
                        self.model.train()
                    elif step_cnt % 300 == 0:
                        self.evaluate(dev_loader)
                        self.model.train()
                        model_name = os.path.join(self.args.save_model_path, "model_{}".format(self.args.savename),
                                                  "step_{}.pkl".format(step_cnt))
                        torch.save(self.model.state_dict(), model_name)
            log_msg = "Epoch average loss is: {}".format(round(epoch_loss / len(train_loader), 4))
            self.logger.info(log_msg)
