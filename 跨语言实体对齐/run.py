from posixpath import join
import torch
import torch.nn as nn
from torch.nn import *
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torch.utils.data as Data
import faiss
import pandas as pd
import argparse
import logging
from datetime import datetime
import torch
from settings import *
from transformers import AutoTokenizer,AutoModel
import pickle
from torch.utils.data import Dataset
from settings import *
import time
# from model_train import Trainer

torch.manual_seed(37)
torch.cuda.manual_seed(37)
np.random.seed(37)

MAX_LEN = 130

def parse_options(parser):
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--time', type=str, default=datetime.now().strftime("%Y%m%d%H%M%S"))
    parser.add_argument('--language', type=str, default='zh_en')
    parser.add_argument('--model_language', type=str, default='zh_en')
    parser.add_argument('--model', type=str, default='LaBSE')

    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--queue_length', type=int, default=64)

    parser.add_argument('--center_norm', type=bool, default=False)
    parser.add_argument('--neighbor_norm', type=bool, default=True)
    parser.add_argument('--emb_norm', type=bool, default=True)
    parser.add_argument('--combine', type=bool, default=True)

    parser.add_argument('--gat_num', type=int, default=1)

    parser.add_argument('--t', type=float, default=0.08)
    parser.add_argument('--momentum', type=float, default=0.9999)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--dropout', type=float, default=0.3)

    return parser.parse_args()

def adjust_learning_rate(optimizer, epoch, lr):
    if (epoch+1) % 10 == 0:
        lr *= 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class DBP15KRawNeighbors():
    def __init__(self, language, doc_id):
        self.language = language
        self.doc_id = doc_id
        self.path = join(DATA_DIR, 'DBP15K', self.language)
        self.id_entity = {}
        self.id_adj_tensor_dict = {}
        self.id_neighbors_dict = {}
        self.load()
        self.id_neighbors_loader()
        self.get_center_adj()

    def load(self):
        with open(join(self.path, "raw_LaBSE_emb_" + self.doc_id + '.pkl'), 'rb') as f:
            self.id_entity = pickle.load(f)

    def id_neighbors_loader(self):
        data = pd.read_csv(join(self.path, 'triples_' + self.doc_id), header=None, sep=' ')
        data.columns = ['head', 'relation', 'tail']

        for index, row in data.iterrows():
            
            head_str = self.id_entity[int(row['head'])][0]
            tail_str = self.id_entity[int(row['tail'])][0]

            if not int(row['head']) in self.id_neighbors_dict.keys():
                self.id_neighbors_dict[int(row['head'])] = [head_str]
            if not tail_str in self.id_neighbors_dict[int(row['head'])]:
                self.id_neighbors_dict[int(row['head'])].append(tail_str)
            
            if not int(row['tail']) in self.id_neighbors_dict.keys():
                self.id_neighbors_dict[int(row['tail'])] = [tail_str]
            if not head_str in self.id_neighbors_dict[int(row['tail'])]:
                self.id_neighbors_dict[int(row['tail'])].append(head_str)
    
    def get_adj(self, valid_len):
        adj = torch.zeros(NEIGHBOR_SIZE, NEIGHBOR_SIZE).bool()
        for i in range(0, valid_len):
            adj[i, i] = 1
            adj[0, i] = 1
            adj[i, 0] = 1
        return adj

    def get_center_adj(self):
        for k, v in self.id_neighbors_dict.items():
            if len(v) < NEIGHBOR_SIZE:
                self.id_adj_tensor_dict[k] = self.get_adj(len(v))
                self.id_neighbors_dict[k] = v + [[0]*LaBSE_DIM] * (NEIGHBOR_SIZE - len(v))
            else:
                self.id_adj_tensor_dict[k] = self.get_adj(NEIGHBOR_SIZE)
                self.id_neighbors_dict[k] = v[:NEIGHBOR_SIZE]

class MyRawdataset(Dataset):
    def __init__(self, id_features_dict, adj_tensor_dict,is_neighbor=True):
        super(MyRawdataset, self).__init__()
        self.num = len(id_features_dict)  # number of samples

        self.x_train = []
        self.x_train_adj = None
        self.y_train = []

        for k in id_features_dict:
            if is_neighbor:
                if self.x_train_adj==None:
                    self.x_train_adj = adj_tensor_dict[k].unsqueeze(0)
                else:
                    self.x_train_adj = torch.cat((self.x_train_adj, adj_tensor_dict[k].unsqueeze(0)), dim=0)
            self.x_train.append(id_features_dict[k])
            self.y_train.append([k])

        self.x_train = torch.Tensor(self.x_train)
        if is_neighbor:
            self.x_train = torch.cat((self.x_train, self.x_train_adj), dim=2)
        self.y_train = torch.Tensor(self.y_train).long()

    # indexing
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return self.num

class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, device, args, n_head=MULTI_HEAD_DIM, f_in=LaBSE_DIM, f_out=LaBSE_DIM, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.device = device
        self.n_head = n_head
        self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(args.dropout)
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        bs, n = h.size()[:2]  # h is of size bs x n x f_in
        h_prime = torch.matmul(h.unsqueeze(1), self.w)  # bs x n_head x n x f_out
        attn_src = torch.matmul(torch.tanh(h_prime), self.a_src)  # bs x n_head x n x 1
        attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst)  # bs x n_head x n x 1
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2)  # bs x n_head x n x n

        attn = self.leaky_relu(attn)
        mask = ~(adj.unsqueeze(1) | torch.eye(adj.shape[-1]).bool().to(self.device))  # bs x 1 x n x n
        attn.data.masked_fill_(mask, float("-inf"))
        attn = self.softmax(attn)  # bs x n_head x n x n
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)  # bs x n_head x n x f_out
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class NCESoftmaxLoss(nn.Module):

    def __init__(self, device):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([batch_size]).to(self.device).long()
        loss = self.criterion(x, label)
        return loss

class MyEmbedder(nn.Module):
    def __init__(self, args, vocab_size, padding=ord(' ')):
        super(MyEmbedder, self).__init__()

        self.args = args
        self.device = torch.device(self.args.device)
        self.attn = BatchMultiHeadGraphAttention(self.device, self.args)        
        self.attn_mlp = nn.Sequential(
            nn.Linear(LaBSE_DIM * 2, LaBSE_DIM),
        )

        # loss
        self.criterion = NCESoftmaxLoss(self.device)

        # batch queue
        self.batch_queue = []

    def contrastive_loss(self, pos_1, pos_2, neg_value):
        bsz = pos_1.shape[0]
        l_pos = torch.bmm(pos_1.view(bsz, 1, -1), pos_2.view(bsz, -1, 1))
        l_pos = l_pos.view(bsz, 1)
        l_neg = torch.mm(pos_1.view(bsz, -1), neg_value.t())
        logits = torch.cat((l_pos, l_neg), dim=1)
        logits = logits.squeeze().contiguous()
        return self.criterion(logits / self.args.t)

    def update(self, network: nn.Module):
        for key_param, query_param in zip(self.parameters(), network.parameters()):
            key_param.data *= self.args.momentum
            key_param.data += (1 - self.args.momentum) * query_param.data
        self.eval()

    def forward(self, batch):
        batch = batch.to(self.device)
        batch_in = batch[:, :, :LaBSE_DIM]
        adj = batch[:, :, LaBSE_DIM:]

        center = batch_in[:, 0].to(self.device)
        center_neigh = batch_in.to(self.device)

        for i in range(0, self.args.gat_num):
            center_neigh = self.attn(center_neigh, adj.bool()).squeeze(1)
        
        center_neigh = center_neigh[:, 0]

        if self.args.center_norm:
            center = F.normalize(center, p=2, dim=1)
        if self.args.neighbor_norm:
            center_neigh = F.normalize(center_neigh, p=2, dim=1)
        if self.args.combine:
            out_hat = torch.cat((center, center_neigh), dim=1)
            out_hat = self.attn_mlp(out_hat)
            if self.args.emb_norm:
                out_hat = F.normalize(out_hat, p=2, dim=1)
        else:
            out_hat = center_neigh

        return out_hat

class Trainer(object):
    def __init__(self, training=True, seed=37):
        # # Set the random seed manually for reproducibility.
        self.seed = seed
        fix_seed(seed)

        # set
        parser = argparse.ArgumentParser()
        self.args = parse_options(parser)

        self.device = torch.device(self.args.device)

        loader1 = DBP15KRawNeighbors(self.args.language, "1")
        myset1 = MyRawdataset(loader1.id_neighbors_dict, loader1.id_adj_tensor_dict)
        del loader1
        
        self.loader1 = Data.DataLoader(
            dataset=myset1,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # all test data
            shuffle=True,
            drop_last=True,
        )

        self.eval_loader1 = Data.DataLoader(
            dataset=myset1,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # all test data
            shuffle=True,
            drop_last=False,
        )

        del myset1

        loader2 = DBP15KRawNeighbors(self.args.language, "2")
        myset2 = MyRawdataset(loader2.id_neighbors_dict, loader2.id_adj_tensor_dict)
        del loader2

        self.loader2 = Data.DataLoader(
            dataset=myset2,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # all test data
            shuffle=True,
            drop_last=True,
        )

        self.eval_loader2 = Data.DataLoader(
            dataset=myset2,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # all test data
            shuffle=True,
            drop_last=False,
        )

        del myset2

        self.model = None
        self.iteration = 0

        # get the linked entity ids
        def link_loader(mode, valid=False):
            link = {}
            if valid == False:
                f = 'test'
            else:
                f = 'valid'
            link_data = pd.read_csv(join(join(DATA_DIR, 'DBP15K', mode), f), sep=' ', header=None)
            link_data.columns = ['entity1', 'entity2']
            entity1_id = link_data['entity1'].values.tolist()
            entity2_id = link_data['entity2'].values.tolist()
            for i, _ in enumerate(entity1_id):
                link[entity1_id[i]] = entity2_id[i]
                link[entity2_id[i]] = entity1_id[i]
            return link

        self.link = link_loader(self.args.language)
        self.val_link = link_loader(self.args.language, True)

        self.neg_queue1 = None
        self.neg_queue2 = None
        self.id_list1 = []
        
    def load_model(self, model_path):
        self.model = MyEmbedder(self.args, VOCAB_SIZE).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def evaluate(self, step):
        logging.info("Evaluate at epoch {}...".format(step))
        print("Evaluate at epoch {}...".format(step))

        ids_1, ids_2, vector_1, vector_2 = list(), list(), list(), list()
        inverse_ids_2 = dict()

        with torch.no_grad():
            self.model.eval()
            for sample_id_1, (token_data_1, id_data_1) in tqdm(enumerate(self.eval_loader1)):
                entity_vector_1 = self.model(token_data_1).squeeze().detach().cpu().numpy()
                ids_1.extend(id_data_1.squeeze().tolist())
                vector_1.append(entity_vector_1)

            for sample_id_2, (token_data_2, id_data_2) in tqdm(enumerate(self.eval_loader2)):
                entity_vector_2 = self.model(token_data_2).squeeze().detach().cpu().numpy()
                ids_2.extend(id_data_2.squeeze().tolist())
                vector_2.append(entity_vector_2)

        for idx, _id in enumerate(ids_2):
            inverse_ids_2[_id] = idx
        def cal_hit(v1, v2, link):
            source = [_id for _id in ids_1 if _id in link]
            target = np.array(
                [inverse_ids_2[link[_id]] if link[_id] in inverse_ids_2 else 99999 for _id in source])
            src_idx = [idx for idx in range(len(ids_1)) if ids_1[idx] in link]
            v1 = np.concatenate(tuple(v1), axis=0)[src_idx, :]
            v2 = np.concatenate(tuple(v2), axis=0)
            index = faiss.IndexFlatL2(v2.shape[1])
            index.add(np.ascontiguousarray(v2))
            D, I = index.search(np.ascontiguousarray(v1), 10)
            hit1 = (I[:, 0] == target).astype(np.int32).sum() / len(source)
            hit10 = (I == target[:, np.newaxis]).astype(np.int32).sum() / len(source)
            logging.info("#Entity: {}".format(len(source)))
            logging.info("Hit@1: {}".format(round(hit1, 3)))
            logging.info("Hit@10:{}".format(round(hit10, 3)))
            print("#Entity: {}".format(len(source)))
            print("Hit@1: {}".format(round(hit1, 3)))
            print("Hit@10:{}".format(round(hit10, 3)))
            return round(hit1, 3), round(hit10, 3)
        
        def cal_mrr(v1, v2, link):
            source = [_id for _id in ids_1 if _id in link and link[_id] in inverse_ids_2]  
            mrr_sum = 0
            total_queries = len(source)
            
            if total_queries == 0:
                logging.warning("No valid links for MRR calculation.")
                print("No valid links for MRR calculation.")
                return 0  
            
            src_idx = [idx for idx in range(len(ids_1)) if ids_1[idx] in source]
            v1 = np.concatenate(tuple(v1), axis=0)[src_idx, :]
            v2 = np.concatenate(tuple(v2), axis=0)
            
            index = faiss.IndexFlatL2(v2.shape[1])
            index.add(np.ascontiguousarray(v2))
            D, I = index.search(np.ascontiguousarray(v1), len(ids_2))
            
            for i, _id in enumerate(source):
                target_idx = inverse_ids_2[link[_id]]  
                rank = np.where(I[i] == target_idx)[0]
                if len(rank) > 0:
                    rank_position = rank[0] + 1
                    mrr_sum += 1 / rank_position
            
            mrr = mrr_sum / total_queries if total_queries > 0 else 0
            return round(mrr, 3)

        logging.info('========Validation========')
        print('========Validation========')
        hit1_valid, hit10_valid = cal_hit(vector_1, vector_2, self.val_link)
        logging.info('===========Test===========')
        print('===========Test===========')
        hit1_test, hit10_test = cal_hit(vector_1, vector_2, self.link)
        mrr_test = cal_mrr(vector_1, vector_2, self.link)
        logging.info("MRR: {}".format(mrr_test))
        print("MRR: {}".format(mrr_test))
        return hit1_valid, hit10_valid, hit1_test, hit10_test,mrr_test

class LaBSEEncoder(nn.Module):
    def __init__(self):
        super(LaBSEEncoder, self).__init__()
        self.device = "cuda:0"
        self.tokenizer = AutoTokenizer.from_pretrained(join(DATA_DIR, "LaBSE"), do_lower_case=False)
        self.model = AutoModel.from_pretrained(join(DATA_DIR, "LaBSE")).to(self.device)

    def forward(self, batch):
        sentences = batch
        tok_res = self.tokenizer(sentences, add_special_tokens=True, padding='max_length', max_length=MAX_LEN)
        input_ids = torch.LongTensor([d[:MAX_LEN] for d in tok_res['input_ids']]).to(self.device)
        token_type_ids = torch.LongTensor(tok_res['token_type_ids']).to(self.device)
        attention_mask = torch.LongTensor(tok_res['attention_mask']).to(self.device)
        output = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return F.normalize(output[0][:, 1:-1, :].sum(dim=1))

class DBP15kRawLoader():
    def __init__(self, language="zh_en", file_suffix="1"):
        self.language = language
        self.file_suffix = file_suffix
        self.id_entity = {}
        self.load()

    def load(self):
        path = join(DATA_DIR, 'DBP15K', self.language)
        file_name = f"cleaned_ent_ids_{self.file_suffix}"
        with open(join(path, file_name), encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                l = line.split(' ')
                id = int(l[0])
                entity = str(l[1])
                self.id_entity[id] = entity

class Embedding(object):
    def __init__(self, language):
        device = "cuda:0"
        self.language = language
        self.loader1 = DBP15kRawLoader(language=self.language, file_suffix="1").id_entity
        self.loader2 = DBP15kRawLoader(language=self.language, file_suffix="2").id_entity
        self.model = LaBSEEncoder().to(device)

    def embedding(self, dir_path):
        id_embedding_1 = {}
        for i, (_id, _ent_name) in tqdm(enumerate(self.loader1.items())):
            emb = self.model([_ent_name]).cpu().detach().numpy().tolist()
            id_embedding_1[int(_id)] = emb
        with open(join(dir_path, "raw_LaBSE_emb_1.pkl"), 'wb') as f:
            pickle.dump(id_embedding_1, f)

        id_embedding_2 = {}
        for i, (_id, _ent_name) in tqdm(enumerate(self.loader2.items())):
            emb = self.model([_ent_name]).cpu().detach().numpy().tolist()
            id_embedding_2[int(_id)] = emb
        with open(join(dir_path, "raw_LaBSE_emb_2.pkl"), 'wb') as f:
            pickle.dump(id_embedding_2, f)

def get_emb():
    device = "cuda:0"
    language_list = ["zh_en"]
    
    for language in language_list:
        embeder = Embedding(language)  
        dir_path = join(DATA_DIR, 'DBP15K', language)
        embeder.embedding(dir_path)  

def result():
    get_emb()
    trainer = Trainer(seed=37)
    model_path = './log2/layers_LaBSE_neighbor/final_model.pth'  
    trainer.load_model(model_path)
    hit1_valid, hit10_valid, hit1_test, hit10_test,mrr_test = trainer.evaluate('0')  
    return hit1_test, hit10_test, mrr_test
