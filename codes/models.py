#!/usr/bin/python3

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator
import collections
from operators import *
from tqdm import tqdm
from util import *
import pdb



class NMP_QEModel(nn.Module):
    def __init__(self, 
                 nentity, 
                 nrelation, 
                 hidden_dim, 
                 gamma,
                 geo,
                 test_batch_size=1,
                 gmm_mode=None, 
                 use_cuda=True,
                 query_name_dict=None,
                 dataset=None):
        super(NMP_QEModel, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.geo = geo
        self.use_cuda = use_cuda
        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1).cuda() if self.use_cuda else torch.arange(nentity).to(torch.float).repeat(test_batch_size, 1) # used in test_step
        self.query_name_dict = query_name_dict

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim
        self.projection_regularizer = Regularizer(1, 0.05, 1e9)


        if dataset == 'NELL':
            self.entity_embedding = nn.Parameter(torch.from_numpy(np.load('path of pretrained embedding')).float())
        if dataset == 'wn18rr':
            self.entity_embedding = nn.Parameter(torch.from_numpy(np.load('path of pretrained embedding')).float())
        if dataset == 'FB15k':
            self.entity_embedding = nn.Parameter(torch.from_numpy(np.load('path of pretrained embedding')).float())
        activation, gmm_num, layers_num = gmm_mode

        self.gmm_num = gmm_num

        self.input_entity_embedding = nn.Parameter(torch.zeros(nentity, 2*gmm_num, hidden_dim+1)) 
        self.init_input(gmm_num, nentity)

        self.projection_regularizer = Regularizer(1, 0.05, 1e9)

        self.projectionNN = RelationProjectionLayer(nrelation, input_dim=self.relation_dim + 1,
                                                    output_dim=self.relation_dim, ngauss=gmm_num, 
                                                    projection_regularizer=self.projection_regularizer)
        
        self.AndNN = AndMLP(nguass=gmm_num, 
                            hidden_dim=hidden_dim, 
                            and_regularizer=self.projection_regularizer)

        self.notNN = NotMLP(n_layers=1, entity_dim=hidden_dim+1)
        # self.OrNN = OrMLP()

    def init_input(self, gmm_num, nentity):
        for i in range(nentity):
            for j in range(gmm_num*2):
                self.input_entity_embedding.data[i, j, :-1] = self.entity_embedding.data[i]
                if j < gmm_num:
                    self.input_entity_embedding.data[i, j, -1] = 1 / gmm_num


    def forward(self, positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict):
        # pdb.set_trace()
        all_center_embeddings, all_idxs = [], []
        all_union_center_embeddings, all_union_idxs = [], []
        for query_structure in batch_queries_dict:
            if 'u' in self.query_name_dict[query_structure]:
                center_embedding, _ = self.embed_query_gmm(self.transform_union_query(batch_queries_dict[query_structure], query_structure),
                                                           self.transform_union_structure(query_structure), 0)
                all_union_center_embeddings.append(center_embedding)
                all_union_idxs.extend(batch_idxs_dict[query_structure])
            else:
                center_embedding, _ = self.embed_query_gmm(batch_queries_dict[query_structure], query_structure, 0)
                all_center_embeddings.append(center_embedding)
                all_idxs.extend(batch_idxs_dict[query_structure])

        if len(all_center_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
        if len(all_union_center_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0] // 2, 2, 1, self.gmm_num*2, -1)

        if type(subsampling_weight) != type(None):
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        if type(positive_sample) != type(None):
            if len(all_center_embeddings) > 0:
                positive_sample_regular = positive_sample[all_idxs]
                positive_embedding = torch.index_select(self.input_entity_embedding, dim=0,
                                                        index=positive_sample_regular).unsqueeze(1)
                positive_logit = self.cal_logit_gmm(positive_embedding, all_center_embeddings)
            else:
                positive_logit = torch.Tensor([]).to(self.input_entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                positive_sample_union = positive_sample[all_union_idxs]
                positive_embedding = torch.index_select(self.input_entity_embedding, dim=0,
                                                        index=positive_sample_union).unsqueeze(1).unsqueeze(1)
                positive_union_logit = self.cal_logit_gmm(positive_embedding, all_union_center_embeddings)
                positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            else:
                positive_union_logit = torch.Tensor([]).to(self.input_entity_embedding.device)
            positive_logit = torch.cat([positive_logit, positive_union_logit], dim=0)
        else:
            positive_logit = None

        if type(negative_sample) != type(None):
            # pdb.set_trace()
            if len(all_center_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]
                batch_size, negative_size = negative_sample_regular.shape
                negative_embedding = torch.index_select(self.input_entity_embedding, dim=0,
                                                        index=negative_sample_regular.view(-1)).view(batch_size, negative_size, self.gmm_num * 2, -1)
                # pdb.set_trace()                    
                negative_logit = self.cal_logit_gmm(negative_embedding, all_center_embeddings)
            else:
                negative_logit = torch.Tensor([]).to(self.input_entity_embedding.device)

            if len(all_union_center_embeddings) > 0:
                negative_sample_union = negative_sample[all_union_idxs]
                batch_size, negative_size = negative_sample_union.shape
                # pdb.set_trace()
                negative_embedding = torch.index_select(self.input_entity_embedding, dim=0,
                                                        index=negative_sample_union.view(-1)).view(batch_size, 1, negative_size, self.gmm_num * 2, -1) # B, 1, neg_size, 2N, dim+1
                negative_embedding = negative_embedding.squeeze(0)
                all_union_center_embeddings = all_union_center_embeddings.squeeze(0)
                negative_union_logit = self.cal_logit_gmm(negative_embedding, all_union_center_embeddings)
                negative_union_logit = negative_union_logit.unsqueeze(0)
                negative_union_logit = torch.max(negative_union_logit, dim=1)[0]
            else:
                negative_union_logit = torch.Tensor([]).to(self.input_entity_embedding.device)
            negative_logit = torch.cat([negative_logit, negative_union_logit], dim=0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, all_idxs + all_union_idxs


########################## for visual ########################################################
    # def embed_query_gmm(self, queries, query_structure, idx):
    #     '''
    #     Iterative embed a batch of queries with same structure using GMM
    #     queries: a flattened batch of queries
    #     '''
    #     # pdb.set_trace()
    #     all_relation_flag = True
    #     for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
    #         if ele not in ['r', 'n']:
    #             all_relation_flag = False
    #             break
    #     # pdb.set_trace()
    #     if all_relation_flag:
    #         if query_structure[0] == 'e':
    #             embedding = torch.index_select(self.input_entity_embedding, dim=0, index=queries[:, idx])
    #             torch.save(embedding, './visual_data/entity_'+ str(idx) + '.pth')
    #             idx += 1
    #         else:
    #             embedding, idx = self.embed_query_gmm(queries, query_structure[0], idx)
    #         for i in range(len(query_structure[-1])): 
    #             if query_structure[-1][i] == 'n':
    #                 # assert False, "gaussian cannot handle queries with negation"
    #                 assert (queries[:, idx] == -2).all()
    #                 embedding = self.notNN(embedding)
    #             else:
    #                 relation_id = queries[:, idx]
    #                 embedding = self.projectionNN(embedding, relation_id)
    #                 torch.save(embedding, './visual_data/p_entity_'+ str(idx) + '.pth')
    #             idx += 1
    #     else:
    #         # queries: 5 * 6, query_structure: (('e', ('r',)), ('e', ('r',)), ('e', ('r',))) embedding_list&offset_embedding_list len: 3 内的元素: 5 * 6400
    #         embedding_list = [] 
    #         for i in range(len(query_structure)):
    #             embedding, idx = self.embed_query_gmm(queries, query_structure[i], idx)
    #             embedding_list.append(embedding)

    #         # pdb.set_trace()
    #         vector = embedding_list[0]
    #         for i in range(1, len(embedding_list)):
    #             vector = self.AndNN(vector, embedding_list[i])
    #             torch.save(vector, './visual_data/insert_entity.pth')
    #         embedding = vector

    #     return embedding, idx

####################################################################################


    def embed_query_gmm(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using GMM
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[-1]: # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        # pdb.set_trace()
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = torch.index_select(self.input_entity_embedding, dim=0, index=queries[:, idx])
                idx += 1
            else:
                embedding, idx = self.embed_query_gmm(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])): 
                if query_structure[-1][i] == 'n':
                    # assert False, "gaussian cannot handle queries with negation"
                    assert (queries[:, idx] == -2).all()
                    embedding = self.notNN(embedding)
                else:
                    relation_id = queries[:, idx]
                    embedding = self.projectionNN(embedding, relation_id)
                idx += 1
        else:
            # queries: 5 * 6, query_structure: (('e', ('r',)), ('e', ('r',)), ('e', ('r',))) embedding_list&offset_embedding_list len
            embedding_list = [] 
            for i in range(len(query_structure)):
                embedding, idx = self.embed_query_gmm(queries, query_structure[i], idx)
                embedding_list.append(embedding)

            # pdb.set_trace()
            vector = embedding_list[0]
            for i in range(1, len(embedding_list)):
                vector = self.AndNN(vector, embedding_list[i])
            embedding = vector

        return embedding, idx


    def transform_union_query(self, queries, query_structure):
        '''
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        '''
        if self.query_name_dict[query_structure] == '2u-DNF':
            queries = queries[:, :-1] # remove union -1
        elif self.query_name_dict[query_structure] == 'up-DNF':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1), torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        queries = torch.reshape(queries, [queries.shape[0]*2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if self.query_name_dict[query_structure] == '2u-DNF':
            return ('e', ('r',))
        elif self.query_name_dict[query_structure] == 'up-DNF':
            return ('e', ('r', 'r'))


    def cal_logit_gmm(self, entity_embedding, query_embedding):
        one_entity_embedding = entity_embedding[:, :, 0, :-1].unsqueeze(-2)
        query_embedding_gauss_prob = query_embedding[:, :, :self.gmm_num, -1]# B, 1(neg_size), N
        query_embedding_mu = query_embedding[:, :, :self.gmm_num, :-1] 
        query_embedding_sigma = query_embedding[:, :, self.gmm_num:, :-1]
        weighted_sigma = torch.matmul(query_embedding_gauss_prob.unsqueeze(-2), query_embedding_sigma)
        weighted_mu = torch.matmul(query_embedding_gauss_prob.unsqueeze(-2), query_embedding_mu) # B, 1(neg_size), 1, N * B, 1(neg_size), N, dim => B, 1(neg_size), 1, dim
        distance = one_entity_embedding - weighted_mu # B, 1(neg_size), 1, dim
        logit = self.gamma - torch.norm(distance, p=1, dim=-1).squeeze(-1) 
        return logit


    @staticmethod 
    def train_step(model, optimizer, train_iterator, args, step):
        model.train()
        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structures = next(train_iterator) # torch.Size([512]) batch_queries:list [[8057, 81, 96, 30], ..] query_structures:list [('e', ('r', 'r', 'r')), ...]
        # pdb.set_trace()
        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        for i, query in enumerate(batch_queries): # group queries with same structure
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        for query_structure in batch_queries_dict:
            if args.cuda:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
            else:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        #with autograd.detect_anomaly():
        if 1==1:
            positive_logit, negative_logit, subsampling_weight, _ = model(positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict)
            # pdb.set_trace()
            asd = F.logsigmoid
            negative_score = asd(-negative_logit).mean(dim=1)
            positive_score = asd(positive_logit).squeeze(dim=1)
            positive_sample_loss = - (subsampling_weight * positive_score).sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()
            positive_sample_loss /= subsampling_weight.sum()
            negative_sample_loss /= subsampling_weight.sum()

            loss = (positive_sample_loss + negative_sample_loss)/2
            loss.backward()
            pdb.set_trace()
            # for name, paras in model.named_parameters():
            #     print('-->name', name, '-->grad_required', paras.requires_grad, '-->grad_value', paras.grad)
            # pdb.set_trace()
            optimizer.step()
        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
        }
        return log

    @staticmethod 
    def test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, save_result=False, save_str="", save_empty=False):
        model.eval()

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader, disable=not args.print_on_screen):
                batch_queries_dict = collections.defaultdict(list)
                batch_idxs_dict = collections.defaultdict(list)
                for i, query in enumerate(queries):
                    batch_queries_dict[query_structures[i]].append(query)
                    batch_idxs_dict[query_structures[i]].append(i)
                for query_structure in batch_queries_dict:
                    if args.cuda:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
                    else:
                        batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
                if args.cuda:
                    negative_sample = negative_sample.cuda()

                # pdb.set_trace()
                _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
                # pdb.set_trace()
                queries_unflatten = [queries_unflatten[i] for i in idxs]
                query_structures = [query_structures[i] for i in idxs]
                argsort = torch.argsort(negative_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float) # [14505,] i.e.[[ 6398.,   268.,  3127.,  ..., 14216., 14504., 14215.]]
                if len(argsort) == args.test_batch_size:
                    ranking = ranking.scatter_(1, argsort, model.batch_entity_range) # achieve the ranking of all entities
                else:
                    if args.cuda:
                        ranking = ranking.scatter_(1,
                                                   argsort,
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                      1).cuda()
                                                   )
                    else:
                        ranking = ranking.scatter_(1,
                                                   argsort,
                                                   torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0],
                                                                                                      1)
                                                   ) # achieve the ranking of all entities
                for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
                    hard_answer = hard_answers[query] # set
                    easy_answer = easy_answers[query] # set
                    num_hard = len(hard_answer)
                    num_easy = len(easy_answer)
                    assert len(hard_answer.intersection(easy_answer)) == 0 
                    cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)] #
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if args.cuda:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                    else:
                        answer_list = torch.arange(num_hard + num_easy).to(torch.float)
                    cur_ranking = cur_ranking - answer_list + 1 # filtered setting
                    cur_ranking = cur_ranking[masks] # only take indices that belong to the hard answers

                    mrr = torch.mean(1./cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                    logs[query_structure].append({
                        'MRR': mrr,
                        'HITS1': h1,
                        'HITS3': h3,
                        'HITS10': h10,
                        'num_hard_answer': num_hard,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int)) 
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]])/len(logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics
