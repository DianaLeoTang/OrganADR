import os
import sys
import json
import argparse
os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.expandvars(r"%LOCALAPPDATA%\torch_extensions"))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")  # 4090 Laptop


from torchdrug.layers.functional import spmm as _spmm
from torchdrug.layers import functional as tdF
tdF.generalized_rspmm = _spmm.generalized_rspmm  # 避免再触发编译
# --------------------------------------------------------------------
print(">>> boot")
import torch
import random
print(">>> torch ok", torch.__version__)
from torchdrug.layers import functional
import pickle
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
from datetime import datetime
import torch.nn.functional as F
import itertools
from sklearn.metrics import accuracy_score, precision_score, recall_score, hamming_loss, roc_auc_score, average_precision_score

os.add_dll_directory(r"C:\Users\tangw\.conda\envs\organadr\Lib\site-packages\torch\lib")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin")

def batch_by_size(batch_size, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])

    start = 0
    while(start < n_sample):
        end = min(n_sample, start + batch_size)
        ret = [ls[start:end] for ls in lists]
        start += batch_size
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]

class DataLoader:
    def __init__(self, args):
        print(">>> dataloader init start")
        # set paramaters
        self.dataset_dir = args.dataset_dir
        self.dataset = args.dataset
        self.args = args
        ddi_paths = {
                'train': os.path.join(self.dataset_dir, '{}/{}/{}/{}.txt'.format(args.dataset, args.datasource, args.dataseed, 'train')),
                'valid': os.path.join(self.dataset_dir, '{}/{}/{}/{}.txt'.format(args.dataset, args.datasource, args.dataseed, 'valid')),
                'test':  os.path.join(self.dataset_dir, '{}/{}/{}/{}.txt'.format(args.dataset, args.datasource, args.dataseed, 'test')),
            }
        kg_path = os.path.join(self.dataset_dir, 'KG1_{}.txt'.format(args.kg))
        
        # load data
        self.process_files_ddi(ddi_paths)
        self.process_files_kg(kg_path)
        
        fact_triplets = []
        for triplet in self.pos_triplets['train']:
            h, t, r = triplet[0], triplet[1], triplet[2:]
            for s in np.nonzero(r)[0]:
                fact_triplets.append([h,t,s])
        self.vtKG = self.load_graph(np.array(fact_triplets), self.kg_triplets)
        self.vtKG = self.vtKG
        print(">>> vtKG ready")
    def process_files_ddi(self, file_paths):
        entity2id = {}
        relation2id = {}

        self.pos_triplets = {}
        self.neg_triplets = {}
        self.train_ent = set()

        for file_type, file_path in file_paths.items():
            pos_triplet = []
            neg_triplet = []
            with open(file_path, 'r') as f:
                for line in f:
                    x, y, z, w = line.strip().split('\t')
                    x, y, w = int(x), int(y), int(w)
                    z1 = list(map(int, z.split(',')))
                    z = [i for i, _ in enumerate(z1) if _ == 1]
                    for s in z:
                        if x not in entity2id:
                            entity2id[x] = x
                        if y not in entity2id:
                            entity2id[y] = y
                        if s not in relation2id:
                            relation2id[s] = s

                    if w==1:
                        pos_triplet.append([x,y] + z1)
                    else:
                        neg_triplet.append([x,y] + z1)
                
                    if file_type == 'train':
                        self.train_ent.add(x)
                        self.train_ent.add(y)
            self.pos_triplets[file_type] = np.array(pos_triplet, dtype='int')
            self.neg_triplets[file_type] = np.array(neg_triplet, dtype='int')

        self.entity2id = entity2id
        self.relation2id = relation2id

        self.eval_rel = len(self.relation2id)

    def process_files_kg(self, kg_path):
        self.kg_triplets = []

        with open(kg_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

            for triplet in file_data:
                h, t, r = int(triplet[0]), int(triplet[1]), int(triplet[2])
                if h not in self.entity2id:
                    self.entity2id[h] = h
                if t not in self.entity2id:
                    self.entity2id[t] = t
                if r not in self.relation2id:
                    self.relation2id[r] = r
                self.kg_triplets.append([h, t, r])

            self.kg_triplets = np.array(self.kg_triplets, dtype='int')

            self.all_ent = max(self.entity2id.keys()) + 1
            self.all_rel = max(self.relation2id.keys()) + 1

    def load_graph(self, triplets, kg_triplets):
        new_triplets = []
        for triplet in triplets:
            h, t, r = triplet
            new_triplets.append([t, h, r])
            new_triplets.append([h, t, r])
        if kg_triplets is not None:
            for triplet in kg_triplets:
                h, t, r = triplet
                r_inv = r + self.all_rel-self.eval_rel
                new_triplets.append([h, t, r])
                new_triplets.append([t, h, r_inv])
        edges = np.array(new_triplets)
        all_rel = 2*self.all_rel - self.eval_rel

        idd = np.concatenate([np.expand_dims(np.arange(self.all_ent),1), np.expand_dims(np.arange(self.all_ent),1), all_rel*np.ones((self.all_ent, 1))],1)
        edges = np.concatenate([edges, idd], axis=0)
        values = np.ones(edges.shape[0])
        adjs = torch.sparse_coo_tensor(indices=torch.LongTensor(edges).t(), 
                                       values=torch.FloatTensor(values), 
                                       size=torch.Size([self.all_ent, self.all_ent, all_rel+1]), requires_grad=False).cuda()
        return adjs

    def shuffle_train(self, ratio=0.8):
        train_ent = set(np.random.choice(list(self.train_ent), int(len(self.train_ent)*ratio), replace=False))
        all_triplet_p = np.array(self.pos_triplets['train'])
        all_triplet_n = np.array(self.neg_triplets['train'])

        if self.dataset.startswith('dataset2'):
            fact_triplet, self.train_pos, self.train_neg = [], [], []
            for i in range(len(all_triplet_p)):
                hp, tp, rp = all_triplet_p[i,0], all_triplet_p[i,1], all_triplet_p[i,2:]
                hn, tn, rn = all_triplet_n[i,0], all_triplet_n[i,1], all_triplet_n[i,2:]
                if hp in train_ent and tp in train_ent:
                    for s in np.nonzero(rp)[0]:
                        fact_triplet.append([hp, tp, s])
                elif (hp in train_ent and tp not in train_ent) or (hp not in train_ent and tp in train_ent):
                    if self.args.trainmethod == 'A':
                        if (hn in train_ent and tn not in train_ent) or (hn not in train_ent and tn in train_ent):
                            self.train_pos.append(self.pos_triplets['train'][i])
                            self.train_neg.append(self.neg_triplets['train'][i])
                    elif self.args.trainmethod == 'B':
                        self.train_pos.append(self.pos_triplets['train'][i])
                        self.train_neg.append(self.neg_triplets['train'][i])

            fact_triplet = np.array(fact_triplet)
            self.train_pos = np.array(self.train_pos)
            self.train_neg = np.array(self.train_neg)
            self.KG = self.load_graph(fact_triplet, self.kg_triplets)

        elif self.dataset.startswith('dataset3'):
            fact_triplet, self.train_pos, self.train_neg = [], [], []
            for i in range(len(all_triplet_p)):
                hp, tp, rp = all_triplet_p[i,0], all_triplet_p[i,1], all_triplet_p[i,2:]
                hn, tn, rn = all_triplet_n[i,0], all_triplet_n[i,1], all_triplet_n[i,2:]
                if hp in train_ent and tp in train_ent:
                    for s in np.nonzero(rp)[0]:
                        fact_triplet.append([hp, tp, s])
                elif hp not in train_ent and tp not in train_ent:
                    if self.args.trainmethod == 'A':
                        if hn not in train_ent and tn not in train_ent:
                            self.train_pos.append(self.pos_triplets['train'][i])
                            self.train_neg.append(self.neg_triplets['train'][i])
                    elif self.args.trainmethod == 'B':
                        self.train_pos.append(self.pos_triplets['train'][i])
                        self.train_neg.append(self.neg_triplets['train'][i])

            fact_triplet = np.array(fact_triplet)
            self.train_pos = np.array(self.train_pos)
            self.train_neg = np.array(self.train_neg)
            self.KG = self.load_graph(fact_triplet, self.kg_triplets)

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))

    def apply_gcn(self, adjacency_matrix, features):
        output = torch.matmul(adjacency_matrix, torch.matmul(features, self.weight))
        return F.relu(output)

class OrganADR(nn.Module):
    def __init__(self, eval_rel, args):
        super(OrganADR, self).__init__()
        self.eval_rel = eval_rel
        self.all_ent = args.all_ent
        self.all_rel = args.all_rel
        self.L = args.length
        self.feature = args.feature
        all_rel = 2*args.all_rel - args.eval_rel + 1
        self.args = args
        self.n_dim = args.n_dim
        self.g_dim = args.g_dim

        # GCN Module
        self.embedding_layer_in = nn.Embedding(num_embeddings=15, embedding_dim=self.g_dim)
        self.embedding_layer_ab = nn.Embedding(num_embeddings=15, embedding_dim=self.g_dim)
        self.gcn_layers = nn.ModuleList([GCNLayer(self.g_dim, self.g_dim) for _ in range(self.L)])
        self.pool = nn.AdaptiveMaxPool1d(1)

        with open('../preprocessed_data/id2drugfeature_new_639.pkl', 'rb') as f:
            x = pickle.load(f, encoding='utf-8')
            feature_01 = []
            feature_02 = []
            feature_03 = [] 
            feature_04 = []
            feature_M = []
            for k in x:
                # Integrate
                feature_01.append(x[k]['RDKit Descriptor'])
                feature_02.append(x[k]['RDKit Fingerprint'])
                feature_03.append(x[k]['MACCS'])
                feature_04.append(x[k]['Morgan_512'])

                # Morgan
                feature_M.append(x[k]['Morgan_1024_raw'])

            feature_01 = np.array(feature_01)
            feature_02 = np.array(feature_02)
            feature_03 = np.array(feature_03)
            feature_04 = np.array(feature_04)
            feature_M = np.array(feature_M)
            feature_I = np.hstack([feature_01, feature_02, feature_03, feature_04])

        if self.feature == 'F':
            self.feature_01 = nn.Parameter(torch.FloatTensor(feature_01), requires_grad=False)
            self.feature_02 = nn.Parameter(torch.FloatTensor(feature_02), requires_grad=False)
            self.feature_03 = nn.Parameter(torch.FloatTensor(feature_03), requires_grad=False)
            self.feature_04 = nn.Parameter(torch.FloatTensor(feature_04), requires_grad=False)
            self.attention_189 = nn.Linear(210, 210)
            self.attention_156 = nn.Linear(166, 166)
            self.Wr1 = nn.Linear(2 * args.n_dim, eval_rel)
            self.Wr2 = nn.Linear(4 * args.n_dim + 16, eval_rel)
            self.Went = nn.Linear(210+136+166+512, args.n_dim)

        elif self.feature == 'M':
            self.ent_kg = nn.Parameter(torch.FloatTensor(feature_M), requires_grad=False)
            self.Went = nn.Linear(1024, args.n_dim)
            self.Wr1 = nn.Linear(2 * args.n_dim, eval_rel)
            self.Wr2 = nn.Linear(4 * args.n_dim + 16, eval_rel)

        self.rel_kg = nn.ModuleList([nn.Embedding(all_rel, args.n_dim) for i in range(self.L)])
        self.linear = nn.ModuleList([nn.Linear(args.n_dim, args.n_dim) for i in range(self.L)])
        
        self.act = nn.ReLU()
        self.relation_linear = nn.ModuleList([nn.Linear(2*args.n_dim, 5) for i in range(self.L)])
        self.attn_relation = nn.ModuleList([nn.Linear(5, all_rel) for i in range(self.L)])
        self.adjacency_matrix = torch.from_numpy(np.load(f'../datasets/{args.dataset}/{args.datasource}/{args.dataseed}/train_mat.npy').astype(np.float32)).cuda()

        self.transfer_matrix = nn.Linear(16, 64)

    def enc_ht(self, head, tail, KG):
        
        if self.feature == 'F':
            head_embed = self.enc_head_or_tail(head)
            tail_embed = self.enc_head_or_tail(tail)
        elif self.feature == 'M' or self.feature == 'I':
            head_embed = self.Went(self.ent_kg[head])
            tail_embed = self.Went(self.ent_kg[tail])

        n_ent = self.all_ent
        
        ht_embed = torch.cat([head_embed, tail_embed], dim=-1)

        hiddens = torch.FloatTensor(np.zeros((n_ent, len(head), self.n_dim))).cuda()
        hiddens[head, torch.arange(len(head)).cuda()] = head_embed
        for l in range(self.L):
            hiddens = hiddens.view(n_ent, -1)
            relation_weight = self.attn_relation[l](nn.ReLU()(self.relation_linear[l](ht_embed)))
            relation_weight = torch.sigmoid(relation_weight).unsqueeze(2)
            rel_embed = self.rel_kg[l].weight
            relation_input = relation_weight * rel_embed
            relation_input = relation_input.view(head_embed.size(0), -1, self.n_dim)
            relation_input = relation_input.transpose(0,1).flatten(1)
            hiddens = functional.generalized_rspmm(KG, relation_input, hiddens, sum='add', mul='mul')
            hiddens = hiddens.view(n_ent * len(head), -1)
            hiddens = self.linear[l](hiddens)
            hiddens = self.act(hiddens)
        tail_hid = hiddens.view(n_ent, len(tail), -1)[tail, torch.arange(len(tail))]

        hiddens = torch.FloatTensor(np.zeros((n_ent, len(head), self.n_dim))).cuda()
        hiddens[tail, torch.arange(len(tail)).cuda()] = tail_embed
        for l in range(self.L):
            hiddens = hiddens.view(n_ent, -1)
            relation_weight = self.attn_relation[l](nn.ReLU()(self.relation_linear[l](ht_embed)))
            relation_weight = torch.sigmoid(relation_weight).unsqueeze(2)
            rel_embed = self.rel_kg[l].weight
            relation_input = relation_weight * rel_embed
            relation_input = relation_input.view(head_embed.size(0), -1, self.n_dim)
            relation_input = relation_input.transpose(0,1).flatten(1)
            hiddens = functional.generalized_rspmm(KG, relation_input, hiddens, sum='add', mul='mul')
            hiddens = hiddens.view(n_ent * len(head), -1)
            hiddens = self.linear[l](hiddens)
            hiddens = self.act(hiddens)
        head_hid = hiddens.view(n_ent, len(head), -1)[head, torch.arange(len(head))]
    
        embeddings = torch.cat([head_hid, tail_hid], dim=1)

        return embeddings
            
    def enc_r1(self, ht_embed):
        scores = self.Wr1(ht_embed)
        return scores
    
    def enc_r2(self, ht_embed):
        scores = self.Wr2(ht_embed)
        return scores

    def enc_head_or_tail(self, head_or_tail):
        weight_210 = self.attention_189(self.feature_01[head_or_tail])
        weight_210 = F.softmax(weight_210, dim=-1)
        embed_210 = self.feature_01[head_or_tail] * weight_210
        weight_166 = self.attention_156(self.feature_03[head_or_tail])
        weight_166 = F.softmax(weight_166, dim=-1)
        embed_166 = self.feature_03[head_or_tail] * weight_166
        embed_136 = self.feature_02[head_or_tail]
        embed_512 = self.feature_04[head_or_tail]
        combined_embed = torch.cat((embed_210, embed_166, embed_136, embed_512), dim=1)
        return self.Went(combined_embed)
    
    def get_label_embedding(self, labels):

        batch_size = labels.shape[0]
        indices = torch.arange(0, 15, device=labels.device).unsqueeze(0).repeat(batch_size, 1)
        selected_embeddings_in = self.embedding_layer_in(indices)
        selected_embeddings_ab = self.embedding_layer_ab(indices)
        embeddings = torch.where(labels.unsqueeze(-1) == 1, selected_embeddings_in, selected_embeddings_ab)

        gcn_output = embeddings
        for gcn in self.gcn_layers:
            gcn_output = gcn.apply_gcn(self.adjacency_matrix, gcn_output)

        pooled_output = self.pool(gcn_output.transpose(1, 2)).squeeze(-1)

        return pooled_output
    
    def attention(self, embed_drug, embed_label):

        LI_transformed = self.transfer_matrix(embed_label)
        attention_scores = torch.mul(embed_drug, LI_transformed)
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_H = attention_weights * embed_drug

        return weighted_H

class BaseModel(object):
    def __init__(self, eval_rel, args, entity_vocab=None, relation_vocab=None):
        self.model = OrganADR(eval_rel, args)
        self.model.cuda()

        self.eval_rel = eval_rel
        self.all_rel = args.all_rel
        self.args = args

        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.lamb)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max')

        self.bce_loss = nn.BCELoss()

        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab

    def train(self, train_pos, train_neg, KG):
        pos_head, pos_tail, pos_label = torch.LongTensor(train_pos[:,0]).cuda(), torch.LongTensor(train_pos[:,1]).cuda(), torch.FloatTensor(train_pos[:,2:]).cuda()
        neg_head, neg_tail, neg_label = torch.LongTensor(train_neg[:,0]).cuda(), torch.LongTensor(train_neg[:,1]).cuda(), torch.FloatTensor(train_neg[:,2:]).cuda()
        n_train = len(pos_head)
        n_batch = self.args.n_batch

        self.model.train()
        for p_h, p_t, p_r, n_h, n_t, n_r in tqdm(batch_by_size(n_batch, pos_head, pos_tail, pos_label, neg_head, neg_tail, neg_label, n_sample=n_train), 
                ncols=100, leave=False, total=len(pos_head)//n_batch+int(len(pos_head)%n_batch>0)):
            self.model.zero_grad()

            p_embs = self.model.enc_ht(p_h, p_t, KG)
            n_embs = self.model.enc_ht(n_h, n_t, KG)

            p_scores1 = torch.sigmoid(self.model.enc_r1(p_embs))
            n_scores1 = torch.sigmoid(self.model.enc_r1(n_embs))
            
            pscores_f, nscores_f = p_scores1, n_scores1

            if self.args.prior_knowledge == 'W':

                p_labels = (p_scores1 > 0.5).int()
                n_labels = (n_scores1 > 0.5).int()

                p_label_embed = self.model.get_label_embedding(p_labels)
                n_label_embed = self.model.get_label_embedding(n_labels)

                weighted_p_drug_embed = self.model.attention(p_embs, p_label_embed)
                weighted_n_drug_embed = self.model.attention(n_embs, n_label_embed)

                concatenated_p = torch.cat((p_embs, p_label_embed, weighted_p_drug_embed), dim=1)
                concatenated_n = torch.cat((n_embs, n_label_embed, weighted_n_drug_embed), dim=1)

                p_scores2 = torch.sigmoid(self.model.enc_r2(concatenated_p))
                n_scores2 = torch.sigmoid(self.model.enc_r2(concatenated_n))
                
                pscores_f, nscores_f = p_scores2, n_scores2

            p_r = p_r.float()

            p_scores = pscores_f[p_r>=0]
            n_scores = nscores_f[p_r>=0]
            scores = torch.cat([p_scores, n_scores], dim=0)
            labels = torch.cat([torch.ones(len(p_scores)), torch.zeros(len(n_scores))], dim=0).cuda()
            loss1 = self.bce_loss(scores, labels)

            p_scores = pscores_f[p_r>0]
            n_scores = nscores_f[p_r>0]
            scores = torch.cat([p_scores, n_scores], dim=0)
            labels = torch.cat([torch.ones(len(p_scores)), torch.zeros(len(n_scores))], dim=0).cuda()
            loss2 = self.bce_loss(scores, labels)

            if self.args.loss == 'loss1':
                loss = loss1
            elif self.args.loss == 'loss2':
                loss = loss2
            elif self.args.loss == 'loss3':
                loss = loss1 + loss2

            loss.backward()
            self.optimizer.step()

    def evaluate(self, test_pos, test_neg, KG):
        pos_head, pos_tail, pos_label = test_pos[:,0], test_pos[:,1], test_pos[:,2:]
        neg_head, neg_tail, neg_label = test_neg[:,0], test_neg[:,1], test_neg[:,2:]
        batch_size = self.args.test_batch_size
        num_batch = len(pos_head) // batch_size + int(len(pos_head)%batch_size>0)

        self.model.eval()
        pos_scores = []
        neg_scores = []
        pred_class = {}
        for i in range(num_batch):
            start = i * batch_size
            end = min((i+1)*batch_size, len(pos_head))
            p_h= pos_head[start:end]
            p_t= pos_tail[start:end]

            n_h= neg_head[start:end]
            n_t= neg_tail[start:end]
            
            p_embs = self.model.enc_ht(p_h, p_t, KG)
            n_embs = self.model.enc_ht(n_h, n_t, KG)

            p_scores1 = torch.sigmoid(self.model.enc_r1(p_embs))
            n_scores1 = torch.sigmoid(self.model.enc_r1(n_embs))

            pscores_f, nscores_f = p_scores1, n_scores1
            
            if self.args.prior_knowledge == 'W':

                p_labels = (p_scores1 > 0.5).int()
                n_labels = (n_scores1 > 0.5).int()

                p_label_embed = self.model.get_label_embedding(p_labels)
                n_label_embed = self.model.get_label_embedding(n_labels)

                weighted_p_drug_embed = self.model.attention(p_embs, p_label_embed)
                weighted_n_drug_embed = self.model.attention(n_embs, n_label_embed)

                concatenated_p = torch.cat((p_embs, p_label_embed, weighted_p_drug_embed), dim=1)
                concatenated_n = torch.cat((n_embs, n_label_embed, weighted_n_drug_embed), dim=1)

                p_scores2 = torch.sigmoid(self.model.enc_r2(concatenated_p))
                n_scores2 = torch.sigmoid(self.model.enc_r2(concatenated_n))

                pscores_f, nscores_f = p_scores2, n_scores2
            
            pos_scores.append(pscores_f.cpu().data.numpy())
            neg_scores.append(nscores_f.cpu().data.numpy())

        labels = pos_label.cpu().data.numpy()
        pos_scores = np.concatenate(pos_scores)
        neg_scores = np.concatenate(neg_scores)
        for r in range(self.eval_rel):
            if self.args.evaltype == 'eval1':
                index = labels[:,r] >= 0
            elif self.args.evaltype == 'eval2':
                index = labels[:,r] > 0

            pred_class[r] = {'score': list(pos_scores[index, r]) + list(neg_scores[index, r]),
                             'preds': list((pos_scores[index, r] > 0.5).astype('int')) + list((neg_scores[index, r] > 0.5).astype('int')),
                             'label': [1] * np.sum(index) + [0] * np.sum(index)}

        roc_auc = []
        prc_auc = []
        accuracy = []
        precision = []
        recall = []
        hamm_loss = []
        for r in range(self.eval_rel):
            label = pred_class[r]['label']
            score = pred_class[r]['score']
            preds = pred_class[r]['preds']
            roc_auc.append(roc_auc_score(label, score))
            prc_auc.append(average_precision_score(label, score))
            accuracy.append(accuracy_score(label, preds))
            precision.append(precision_score(label, preds))
            recall.append(recall_score(label, preds))
            hamm_loss.append(hamming_loss(label, preds))

        return np.mean(roc_auc), np.mean(prc_auc), np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(hamm_loss), \
               roc_auc, prc_auc, accuracy, precision, recall, hamm_loss, pred_class

if __name__ == '__main__':
    print(">>> parse config ok:", args_input.config)
    parser = argparse.ArgumentParser(description="Load a configuration file.")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args_input = parser.parse_args()

    class DotDict:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def __setattr__(self, key, value):
            super().__setattr__(key, value)

        def __delattr__(self, item):
            if item in self.__dict__:
                super().__delattr__(item)
            else:
                print(f"{item} does not exist")

    with open(args_input.config, 'r') as file:
        config = json.load(file)

    args = DotDict(**config['args'])

    # data
    list_of_dataset = config['list_of_dataset']
    list_of_datasource = config['list_of_datasource']
    list_of_dataseed = config['list_of_dataseed']

    # knowledge graph
    list_of_kg = config['list_of_kg']

    # train
    list_of_trainseed = config['list_of_trainseed']
    list_of_trainmethod = config['list_of_trainmethod']

    # model
    list_of_feature = config['list_of_feature']
    list_of_loss = config['list_of_loss']
    list_of_evaltype = config['list_of_evaltype']
    list_of_prior_knowledge = config['list_of_prior_knowledge']

    combinations = itertools.product(
        list_of_dataset, list_of_datasource, list_of_dataseed, 
        list_of_trainseed, list_of_trainmethod, list_of_feature, list_of_loss, 
        list_of_kg, list_of_evaltype, list_of_prior_knowledge
    )

    for dataset, datasource, dataseed, trainseed, trainmethod, feature, loss, kg, evaltype, prior_knowledge in combinations:
        
        if dataset == 'dataset2':
            args.lamb = 0.000001
            args.n_dim = 32
            args.g_dim = 16
            args.lr = 0.001
        elif dataset == 'dataset3':
            args.lamb = 0.000001
            args.n_dim = 32
            args.g_dim = 16
            args.lr = 0.001

        args.trainseed = trainseed
        args.kg = kg
        args.dataseed = dataseed
        args.trainmethod = trainmethod
        args.feature = feature
        args.loss = loss
        args.dataset = dataset
        args.datasource = datasource
        args.evaltype = evaltype
        args.prior_knowledge = prior_knowledge

        os.environ['PYTHONHASHSEED'] = str(args.trainseed)
        random.seed(args.trainseed)
        np.random.seed(args.trainseed)
        torch.manual_seed(args.trainseed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.cuda.manual_seed_all(args.trainseed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

        # --------------------
        # torch.use_deterministic_algorithms(True, warn_only=True).
        # RuntimeError: adaptive_max_pool2d_backward_cuda does not have a deterministic implementation.
        # --------------------

        torch.cuda.set_device(args.gpu)

        dataloader = DataLoader(args)
        print(">>> dataloader ok")
        eval_rel = dataloader.eval_rel
        args.all_ent, args.all_rel, args.eval_rel = dataloader.all_ent, dataloader.all_rel, dataloader.eval_rel
        vtKG = dataloader.vtKG
        pos_triplets, neg_triplets = dataloader.pos_triplets, dataloader.neg_triplets
        valid_pos, valid_neg = torch.LongTensor(pos_triplets['valid']).cuda(), torch.LongTensor(neg_triplets['valid']).cuda()
        test_pos,  test_neg  = torch.LongTensor(pos_triplets['test']).cuda(),  torch.LongTensor(neg_triplets['test']).cuda()

        args.train_ent = list(dataloader.train_ent)

        if not os.path.exists('../results/demo'):
            os.makedirs('../results/demo')

        model = BaseModel(eval_rel, args)
        print(">>> model ok")
        now = datetime.now()
        begin_time = now.strftime("%Y_%m_%d_%H_%M_%S")
        filepath = f'../results/demo/{args.dataset}_{args.datasource}_{args.kg}_{args.feature}_{args.prior_knowledge}_{args.trainmethod}_{args.loss}_{args.evaltype}_{args.dataseed}_{args.trainseed}_{begin_time}'
        filename = f'ndim={args.n_dim}_lr={args.lr}_bs={args.n_batch}_lamb={args.lamb}_gdim={args.g_dim}.txt'

        if not os.path.exists(filepath):
            os.makedirs(filepath)

        for e in range(args.n_epoch):
            if args.trainmethod == 'A':
                dataloader.shuffle_train(0.5)
            else:
                dataloader.shuffle_train(0.8)
            KG = dataloader.KG
            train_pos, train_neg = dataloader.train_pos, dataloader.train_neg
            model.train(train_pos, train_neg, KG)
            if (e+1) % args.epoch_per_test == 0:
                v_roc, v_pr, v_acc, v_prec, v_rec, v_hl, _, _, _, _, _, _, v_pred_class = model.evaluate(valid_pos, valid_neg, vtKG)
                t_roc, t_pr, t_acc, t_prec, t_rec, t_hl, o_roc_auc, o_pr_auc, o_acc, o_prec, o_rec, o_hl, t_pred_class = model.evaluate(test_pos, test_neg, vtKG)

                out_str = f'epoch:{e+1:>3d}\t [Valid] PR-AUC:{v_pr:>9.6f} ROC-AUC:{v_roc:>9.6f} ACC:{v_acc:>9.6f} PREC:{v_prec:>9.6f} REC:{v_rec:>9.6f} HL:{v_hl:>9.6f}\t ' \
                        f'[Test] PR-AUC:{t_pr:>9.6f} ROC-AUC:{t_roc:>9.6f} ACC:{t_acc:>9.6f} PREC:{t_prec:>9.6f} REC:{t_rec:>9.6f} HL:{t_hl:>9.6f}'

                print_str = f'epoch:{e+1:>3d}\t [Valid] PR-AUC:{v_pr:>9.6f} ROC-AUC:{v_roc:>9.6f}\t [Test] PR-AUC:{t_pr:>9.6f} ROC-AUC:{t_roc:>9.6f}'
                print(print_str)

                with open(os.path.join(filepath, filename), 'a+') as f:
                    f.write(out_str + '\n')

                with open(f'{filepath}/v_result_{e+1}.pkl', 'wb') as f:
                    pickle.dump(v_pred_class, f)
                with open(f'{filepath}/t_result_{e+1}.pkl', 'wb') as f:
                    pickle.dump(t_pred_class, f)

                torch.save({
                    'model_state_dict': model.model.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                }, f'{filepath}/model_{e+1}.pth')