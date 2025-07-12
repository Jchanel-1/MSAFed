import copy
import logging
import random
import sys
import os
from typing import Collection

import numpy as np
import pandas as pd
import torch
from .client import Client
from tqdm import tqdm
from tensorboardX import SummaryWriter
from copy import deepcopy
import torch.nn as nn
from tensorboardX import SummaryWriter
import collections
from .Queue import Memory
from numpy.linalg import norm
import gc

class FederatedExecuter(nn.Module):
    def __init__(self, dataset, device, args, model_trainer, adapt_lrs=None, train_len=None):
        """
        dataset: data loaders and data size info
        """
        super(FederatedExecuter, self).__init__()
        self.device = device
        self.args = args
        client_num, [train_data_num, val_data_num, test_data_num, train_data_local_num_dict,
                     train_data_local_dict, val_data_local_dict, test_data_local_dict, ood_data] = dataset
        self.client_num_in_total = client_num
        self.client_num_per_round = int(self.client_num_in_total) 
        self.train_data_num_in_total = train_data_num  
        self.val_data_num_in_total = val_data_num  
        self.test_data_num_in_total = test_data_num 

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict  
        self.train_data_local_dict = train_data_local_dict
        self.val_data_local_dict = val_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.ood_data = ood_data  

        self.model_trainer = model_trainer
        # setup clients
        self._setup_clients(train_data_local_num_dict, train_data_local_dict,
                            val_data_local_dict, test_data_local_dict, model_trainer, train_len=train_len)
        logging.info("############setup ood clients#############")
        self.ood_client = Client(-1, None, None, ood_data,
                                 len(ood_data.dataset), self.args, self.device, model_trainer)

        self.local_performance_by_global_model = dict() 
        self.local_val_by_global_model = dict()   
        for idx in range(client_num): 
            self.local_performance_by_global_model[f'idx{idx}'] = [] 
            self.local_val_by_global_model[f'idx{idx}'] = []

        if args.pretrain: 
            self.writer_val_acc={}
            self.writer_val_loss={}
            self.writer_test_acc={}
            self.writer_train_acc={}
            self.writer_train_loss={}
            self.base_path=args.base_path 
            for client_name in self.args.source:
                base=os.path.join(self.base_path,'data_flow',str(client_name))
                if not os.path.exists(base):
                    os.makedirs(base)
                    print('create the writer path for base:{}'.format(base))
                val_path=os.path.join(base,'val')
                if not os.path.exists(val_path):
                    os.makedirs(val_path)
                    print('create the writer path for val_path:{}'.format(val_path))      
                self.writer_val_acc.update({client_name:SummaryWriter(val_path)})
                self.writer_val_loss.update({client_name:SummaryWriter(val_path)})
                train_path=os.path.join(base,'train')
                if not os.path.exists(train_path):
                    os.makedirs(train_path)
                    print('create the writer path for train_path:{}'.format(train_path))
                self.writer_train_loss.update({client_name:SummaryWriter(train_path)})
                self.writer_train_acc.update({client_name:SummaryWriter(train_path)})
                test_path=os.path.join(base,'test')
                if not os.path.exists(test_path):
                    os.makedirs(test_path)
                    print('create the writer path for test_path:{}'.format(test_path))
                self.writer_test_acc.update({client_name:SummaryWriter(test_path)})    


        self.writer = SummaryWriter(log_dir=args.tensor_log)
        if args.train :
            base_path = os.path.join(args.tensor_log, "ada_see")
            self.writer_ada = {}
            self.writer_val = {}
            self.writer_test = {}
            for idx,client_name in enumerate(self.args.source):
                self.writer_ada.update(
                    {idx: SummaryWriter(os.path.join(base_path, 'training', str(client_name)))})
                self.writer_val.update(
                    {idx: SummaryWriter(os.path.join(base_path, 'validation', str(client_name)))})
                self.writer_test.update(
                    {idx: SummaryWriter(os.path.join(base_path, 'test', str(client_name)))})
        self.server_Q=Memory(num_sample=args.num_cluster,args=args,mode="server") ##server queue
        if args.ood_test or args.test:
            base_path=self.args.base_path
            self.writer_test = {}
            for idx,client_name in enumerate(self.args.source):
                self.writer_test.update(
                    {idx: SummaryWriter(os.path.join(base_path, 'test', str(client_name)))})                

        if adapt_lrs == None:
            self.adapt_lrs = torch.zeros(
                len(self.trainable_parameters())).to(self.device)  
        else:
            self.adapt_lrs = copy.deepcopy(adapt_lrs).to(self.device)
        print(len(self.trainable_parameters()))  
        self.new_weights = collections.OrderedDict()
        self.bn_var_idx = self.get_mean_var()  

    def get_mean_var(self):
        names = self.get_name_trainable_parameters()
        mean_var = list()
        for idx, name in enumerate(names):
            if 'running_var' in name or 'running_mean' in name:
                mean_var.append(idx)
        return mean_var

    def trainable_parameters(self):
        ps = list()
        for name, param in self.named_parameters():
            ps.append(param)
  

        return ps 


    def get_name_trainable_parameters(self):
        names = list()
        for name, param in self.named_parameters():
            names.append(name)
        return names  

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, val_data_local_dict, test_data_local_dict, model_trainer, train_len=None):
        logging.info("############setup inner clients#############")
        for client_idx in range(self.client_num_in_total):
            c = Client(client_idx, train_data_local_dict[client_idx], val_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer, train_len=train_len) 
            self.client_list.append(c)


    def adapt_and_eval(self, args, rnd=None, mode='val'): 
        logging.info(
            "============using global model: local_validation_on_all_clients ")

        global_state = deepcopy(
            self.model_trainer.get_model_params()) 
        weights = []  
        losses = []  
        metrics = {'acc': [], 'losses': []}  
        if args.verbose:  
            print(self.adapt_lrs)

        if mode == 'val':
            logging.info('===================validation======================')
            for client_idx in range(self.client_num_in_total):
                if self.val_data_local_dict[client_idx] is None:
                    continue
                client = self.client_list[client_idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.val_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])
                local_metrics = client.local_validate()  

                self.local_val_by_global_model["idx" + str(client_idx)].append(
                    copy.deepcopy(local_metrics['test_acc']))
        
                metrics['acc'].append(copy.deepcopy(local_metrics['test_acc']))
                metrics['losses'].append(copy.deepcopy(
                    local_metrics['test_loss']))  

                logging.info('Client Index = {}\tAcc:{:.4f}\tLoss: {:.4f}'.format(
                    client_idx, local_metrics['test_acc'], local_metrics['test_loss']))
                self.writer_val[client_idx].add_scalar(
                    'each_client_val_acc', local_metrics['test_acc'], rnd)

            avg_val_acc = sum(metrics['acc'])/len(metrics['acc'])
            avg_val_loss = sum(metrics['losses'])/len(metrics['losses'])

            self.writer.add_scalar('val_acc', avg_val_acc, rnd)
            self.writer.add_scalar('val_loss', avg_val_loss, rnd)

            return avg_val_acc


        if mode == 'test':
            logging.info('===================test======================')
            for client_idx,client_name in enumerate(self.args.source):
                if self.test_data_local_dict[client_idx] is None:
                    continue
                client = self.client_list[client_idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.val_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])
                local_metrics = client.local_test(True)
                self.local_val_by_global_model["idx" + str(client_idx)].append(
                    copy.deepcopy(local_metrics['test_acc']))
                metrics['acc'].append(copy.deepcopy(local_metrics['test_acc']))
                logging.info('Client Index = {}\tAcc:{:.4f}\tLoss: {:.4f}'.format(
                    client_idx, local_metrics['test_acc'], local_metrics['test_loss']))
                self.writer_test[client_idx].add_scalar(
                    'each_client_test_acc', local_metrics['test_acc'], rnd)

            if rnd != None:
                avg_test_acc = sum(metrics['acc'])/len(metrics['acc'])
                self.writer.add_scalar('test_acc', avg_test_acc, rnd)

            return metrics  

        if mode == 'ood':
            logging.info('===================ood======================')
            client = self.ood_client
            ood_metrics = client.local_test(True)
            logging.info('ood_client:{},acc={},loss={}'.format(
                args.target, ood_metrics['test_acc'], ood_metrics['test_loss']))

    def run(self, args):
        logging.info("No Adaptation")
        self.adapt_and_eval(args, mode='test')  
        avg_val_score = dict() 
        avg_val_score['acc'] = -1000
        for rnd in range(1, args.comm_round+1):
            tqdm.write('Round:%d /%d ' % (rnd, args.comm_round))
            self.learn_to_adapt(args, args.mode_1,queue=self.server_Q,rnd=rnd) ##mode_1
            val_acc = self.adapt_and_eval(args, rnd, mode='val')
            if avg_val_score['acc'] < val_acc:
                avg_val_score['acc'] = val_acc
                m0=[]
                m1=[]
                for idx, client in enumerate(self.client_list):
                    prototype = client.queue.store_prototype()
                    for point_id in range(self.args.num_cluster):
                        m0.append(torch.tensor(prototype[0][point_id]))
                        m1.append(torch.tensor(prototype[1][point_id]))
                m0=torch.stack(m0)
                m1=torch.stack(m1)
                self.server_Q._ini_test(m0,0)
                self.server_Q._ini_test(m1,1)
                if args.save_weights and args.save_lr:
                    torch.save(self.adapt_lrs, args.save_lr_path)
                    torch.save(self.new_weights, args.save_weights_path)
                    print("########################################################")
                    print(
                        "in the {} communication rounds,performance improves,model be saved".format(rnd))
                    print("########################################################")
                if args.save_prototype:
                    m0 = []
                    m1 = []
                    for idx, client in enumerate(self.client_list):
                        prototype = client.queue.store_prototype()
                        for point_id in range(self.args.num_cluster):
                            m0.append(prototype[0][point_id])
                            m1.append(prototype[1][point_id])
                    torch.save(m0,os.path.join(self.args.save_prototype_path,"m0.pt"))
                    torch.save(m1,os.path.join(self.args.save_prototype_path,"m1.pt"))
            m0 = []
            m1 = []
            for idx, client in enumerate(self.client_list):
                prototype = client.queue.store_prototype()
                for point_id in range(self.args.num_cluster):
                    m0.append(prototype[0][point_id])
                    m1.append(prototype[1][point_id])

            

            test_metrics = self.adapt_and_eval(args, rnd, mode='test')

        logging.info("Test result")
        self.adapt_and_eval(args, mode='test')


    def aggreate_weights(self, args, client_id):
        if client_id == 0:  
            self.new_weights = self.model_trainer.get_model_params()
            for key in self.new_weights:
                self.new_weights[key] = self.new_weights[key]*(
                    self.train_data_local_num_dict[client_id]/self.train_data_num_in_total)
        else:
            ww = self.model_trainer.get_model_params()
            for key in ww:
                self.new_weights[key] += ww[key]*(
                    self.train_data_local_num_dict[client_id]/self.train_data_num_in_total)

    def learn_to_adapt(self, args, mode,queue=None, rnd=None):
        """
        Use training clients' validation data to learn to adapt
        """
        weights = []
        losses = [] 
        metrics = [] 
        if rnd==1:
            global_state = deepcopy(self.model_trainer.get_model_params())
        else :
            global_state = deepcopy(self.new_weights)
        global_adapt_lrs = deepcopy(self.adapt_lrs)
        accum_adapt_lrs = torch.zeros_like(self.adapt_lrs)
        logging.info("============ Communication round : {}".format(rnd))
        client_indexes = self._client_sampling(rnd, self.client_num_in_total,
                                               self.client_num_per_round)
        logging.info("client_indexes = " + str(client_indexes))
        for idx, client in enumerate(self.client_list):
            client_idx = client_indexes[idx] 
            client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                        self.val_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])  

            loss, metric, num_data = client.local_train(
                self.model_trainer, self.adapt_lrs, args, mode,Squeue=queue)  
            accum_adapt_lrs += self.adapt_lrs  
            if args.is_vis_adalr:
                ada_sum = torch.mean(self.adapt_lrs)
                self.writer_ada[idx].add_scalar('ada', ada_sum, rnd)

            self.adapt_lrs.copy_(global_adapt_lrs)  
            if args.save_weights:  
                self.aggreate_weights(args, idx)  
            self.model_trainer.load_state_dict(global_state, strict=False)
            weights.append(num_data)
            losses.append(loss)
            metrics.append(metric)

        self.adapt_lrs = accum_adapt_lrs/len(self.client_list)
        param_name_list = self.get_name_trainable_parameters()

        agg_loss = sum([weight * loss for weight,
                       loss in zip(weights, losses)]) / sum(weights)
        agg_metric = sum([weight * metric for weight,
                         metric in zip(weights, metrics)]) / sum(weights)
        print("agg_loss:{}".format(agg_loss))
        print("agg_metirc:{}".format(agg_metric))
        if rnd != None:
            self.writer.add_scalar('train_agg_loss', agg_loss, rnd)
            self.writer.add_scalar('train_agg_metric', agg_metric, rnd)
        tqdm.write('\t Train:  Loss: %.4f \t Metric: %.4f' %
                   (agg_loss, agg_metric))


    def train(self):
        w_global = self.model_trainer.get_model_params() 

        val_scores = dict()  
        avg_val_score=dict()

        for i in range(self.client_num_in_total):
            val_scores['{}'.format(i)] = -1000
        avg_val_score['acc']=-1000

        for round_idx in range(self.args.comm_round):

            logging.info(
                "============ Communication round : {}".format(round_idx))

            w_locals = []
            prototypes=[]


            client_indexes = self._client_sampling(round_idx, self.client_num_in_total,
                                                   self.client_num_per_round)
            logging.info("client_indexes = " + str(client_indexes))
            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]  
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.val_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])  

   
                w = client.train(copy.deepcopy(w_global),Squeue=self.server_Q,rnd=round_idx) 
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            m0=[]
            m1=[]
            for idx, client in enumerate(self.client_list):
                prototype = client.queue.store_prototype()
                for point_id in range(self.args.num_cluster):
                    m0.append(torch.tensor(prototype[0][point_id]))
                    m1.append(torch.tensor(prototype[1][point_id]))
            
            fea_weights=self.val_heter_pro(m0,m1)
            w_global = self._aggregatev2(fea_weights,w_locals)
            self.model_trainer.set_model_params(w_global)

            # local validation

            val_metrics = self._local_val_on_all_clients(round_idx)  
            avg_metric_val_acc=(sum(val_metrics['acc'])/len(val_metrics['acc']))
            avg_metric_val_loss=(sum(val_metrics['losses'])/len(val_metrics['losses']))
            logging.info("average val performance:{:.4f}".format(avg_metric_val_acc))
            logging.info("average val loss:{:.4f}".format(avg_metric_val_loss))


            if avg_val_score['acc']<avg_metric_val_acc:
                logging.info('in {} rounds,global model improves'.format(round_idx))
                avg_val_score['acc']=avg_metric_val_acc
                ###update the server memory
                m0=torch.stack(m0)
                m1=torch.stack(m1)
                self.server_Q._ini_test(m0,0)
                self.server_Q._ini_test(m1,1)
                if self.args.save_global_weights:
                    torch.save(w_global,self.args.save_global_path) 

            test_metrics=self._local_test_on_all_clients(round_idx) 
            avg_metric_test=(sum(test_metrics['acc'])/len(test_metrics['acc']))
            logging.info("average test performance:{:.4f}".format(avg_metric_test))            


            self.writer.add_scalar('avg_val_acc',avg_metric_val_acc,round_idx)
            self.writer.add_scalar('avg_test_acc',avg_metric_test,round_idx)


            for idx,client_name in enumerate(self.args.source):
                self.writer_val_acc[client_name].add_scalar('val_acc',val_metrics['acc'][idx],round_idx)
                self.writer_val_loss[client_name].add_scalar('val_loss',val_metrics['losses'][idx],round_idx)
                self.writer_test_acc[client_name].add_scalar('test_acc',test_metrics['acc'][idx],round_idx)


    def val_heter_pro(self,m0,m1):
        m0=torch.stack(m0)
        m1=torch.stack(m1)
        M=torch.cat((m0,m1))
        M=np.array(M)
        client_num=len(self.args.source)
        step=self.args.num_cluster
        local_prototype=np.zeros((client_num,self.args.num_class,64))
        for ii in range(client_num):
            for kk in range(self.args.num_class):
                    start_point=step*ii+kk*client_num*step 
                    local_prototype[ii,kk,...]=np.sum(M[start_point:start_point+step,...],axis=0)/self.args.num_cluster
        global_prototype=np.sum(local_prototype,axis=0)/client_num
        similarity=np.zeros((client_num,self.args.num_class))
        for ii in range (client_num):
            s=np.sum(global_prototype*local_prototype[ii],axis=1)/(norm(global_prototype,axis=-1)*norm(local_prototype[ii].T,axis=0))
            similarity[ii]=s/0.01
        attention_weight=np.exp(similarity)/np.sum(np.exp(similarity),axis=0)
        attention_weight=attention_weight/0.1 ##sharpen
        b=np.sum(attention_weight,axis=1)
        weights=b/np.sum(b)
        print(weights)
        return weights    

    def _aggregatev2(self,w_f, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx] 
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w=(1-self.args.agg_coe)*(local_sample_number / training_num)+self.args.agg_coe*w_f[i]
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        
        ##just for print
        for i in range(0, len(w_locals)):
            local_sample_number, local_model_params = w_locals[i]
            w = 0.9*local_sample_number / training_num+0.1*w_f[i]
            print("final:w[{}]={}".format(i,w))
            print("naive:w[{}]={}".format(i,local_sample_number / training_num))
        
        return averaged_params    


    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(
                client_num_in_total)]  
        else:
            num_clients = min(client_num_per_round,
                              client_num_in_total)
            np.random.seed(round_idx)
            client_indexes = np.random.choice(range(
                client_num_in_total), num_clients, replace=False)  
        logging.info("client_indexes = %s" %
                     str(client_indexes))  
        return client_indexes

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx] 
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params


    def _local_val_on_all_clients(self, round_idx):
        logging.info(
            "============ local_validation_on_all_clients : {}".format(round_idx))

        val_metrics = {
            'acc': [],
            'losses': []
        }

        for client_idx in range(self.client_num_in_total):
            if self.val_data_local_dict[client_idx] is None:
                continue
            client = self.client_list[client_idx]
            client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                        self.val_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            local_metrics = client.local_validate()
            self.local_val_by_global_model["idx" + str(client_idx)].append(
                copy.deepcopy(local_metrics['test_acc']))
            val_metrics['acc'].append(copy.deepcopy(local_metrics['test_acc']))
            val_metrics['losses'].append(copy.deepcopy(local_metrics['test_loss']))
            logging.info('Client Index = {}\tAcc:{:.4f}\tLoss: {:.4f}'.format(
                client_idx, local_metrics['test_acc'], local_metrics['test_loss']))
        return val_metrics  

    def _local_test_on_all_clients(self, round_idx):
        logging.info(
            "============ local_test_on_all_clients : {}".format(round_idx))

        test_metrics = {
            'acc': [],
            'losses': []
        }

        for client_idx in range(self.client_num_in_total):
            if self.test_data_local_dict[client_idx] is None:
                continue
            client = self.client_list[client_idx]
            client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                        self.val_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            # test data
            test_local_metrics = client.local_test(True)

            self.local_performance_by_global_model["idx" + str(client_idx)].append(
                copy.deepcopy(test_local_metrics['test_acc']))
            test_metrics['acc'].append(
                copy.deepcopy(test_local_metrics['test_acc']))
            test_metrics['losses'].append(
                copy.deepcopy(test_local_metrics['test_loss']))
            logging.info('Client Index = {}\tAcc:{:.4f}\tLoss: {:.4f}'.format(
                client_idx, test_local_metrics['test_acc'], test_local_metrics['test_loss']))
        return test_metrics

