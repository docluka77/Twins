#!/usr/bin/env python
#title           :trainers.py
#description     :This file includes the trainers of hw2vec.
#author          :Shih-Yuan Yu
#date            :2021/03/05
#version         :0.2
#notes           :
#python_version  :3.6
#==============================================================================
import os, sys
sys.path.append(os.path.dirname(sys.path[0]))
import matplotlib.pyplot as plt
import pandas as pd

import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle as pkl
import csv
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.data import Data, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
from torch_geometric.utils import scatter
from hw2vec.graph2vec.models import *
from hw2vec.utilities import *

from pytorch_metric_learning import distances, losses, miners, reducers

class BaseTrainer:
    def __init__(self, cfg):
        self.config = cfg
        self.min_test_loss = np.Inf
        self.task = None
        self.metrics = {}
        self.model = None
        self.loss_value = []
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    def build(self, model, path=None):
        self.model = model
        # self.optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=5e-4,eps=1e-3)
        self.optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=5e-3)
        # self.optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[100, 200, 300], gamma=0.5
        )
        self.reducer = ExponentialLR(self.optimizer, gamma=0.95)
        self.optimizer1 = optim.Adam(model.parameters(), lr=self.config.learning_rate*2, weight_decay=5e-3)
        
    def visualize_embeddings(self, data_loader, path=None):
        save_path = "./visualize_embeddings/" if path is None else Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        embeddings, hw_names = self.get_embeddings(data_loader)

        with open(str(save_path / "vectors.tsv"), "w") as vectors_file, \
             open(str(save_path / "metadata.tsv"), "w") as metadata_file:

            for embed, name in zip(embeddings, hw_names):
                vectors_file.write("\t".join([str(x) for x in embed.detach().cpu().numpy()[0]]) + "\n")
                metadata_file.write(name+"\n")

    def get_embeddings(self, data_loader):
        embeds = []
        hw_names = []

        with torch.no_grad():
            self.model.eval()

            for data in data_loader:
                data.to(self.config.device)
                embed_x, _ = self.model.embed_graph(data.x, data.edge_index, data.batch)
                embeds.append(embed_x)
                hw_names += data.hw_name

        return embeds, hw_names

    def metric_calc(self, loss, labels, preds, header):
        '''计算结果'''
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="binary")
        conf_mtx = str(confusion_matrix(labels, preds)).replace('\n', ',')
        precision = precision_score(labels, preds, average="binary")
        recall = recall_score(labels, preds, average="binary")

        self.metric_print(loss, acc, f1, conf_mtx, precision, recall, header)

        if header == "test " and (self.min_test_loss >= loss):
            self.min_test_loss = loss
            self.metrics["acc"] = acc
            self.metrics["f1"] = f1
            self.metrics["conf_mtx"] = conf_mtx
            self.metrics["precision"] = precision
            self.metrics["recall"] = recall

    def metric_print(self, loss, acc, f1, conf_mtx, precision, recall, header):
        print("%s loss: %4f" % (header, loss) +
            ", %s accuracy: %.4f" % (header, acc) +
            ", %s f1 score: %.4f" % (header, f1) +
            ", %s confusion_matrix: %s" % (header, conf_mtx) +
            ", %s precision: %.4f" % (header, precision) +
            ", %s recall: %.4f" % (header, recall))

class PairwiseGraphTrainer(BaseTrainer):
    ''' trainer for graph classification ''' 
    def __init__(self, cfg):
        super().__init__(cfg)
        self.task = "IP"
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-08).to(self.config.device)
        self.cos_loss = torch.nn.CosineEmbeddingLoss(margin=0.5).to(self.config.device)
        self.mse_loss = torch.nn.MSELoss(reduction='sum').to(self.config.device)
        self.log_loss = torch.nn.CrossEntropyLoss()
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
    
    def draw_loss(self,lossfile,savefile = 'my_plot.png'):
      # 读取csv文件并转换成DataFrame格式
      lossfile = os.path.join("losspng",lossfile)
      savefile = os.path.join("losspng",savefile)
      data = pd.read_csv(lossfile)
      
      # 获取x和y列的数据
      x = data.iloc[:, 0]
      y = data.iloc[:, 1]
      
      # 用Matplotlib画折线图
      plt.plot(x, y)
      
      # 添加标题和标签
      plt.title(savefile)
      plt.xlabel('x')
      plt.ylabel('y')
      
      # 显示图形
    #   plt.show()  
      plt.savefig(savefile)
    
    def train(self, train_loader, test_loader):
        tqdm_bar = tqdm(range(self.config.epochs))
        # with open('train_loss.csv')as f:
        # with open('train_loss.csv', 'w') as f:s
        # tsv = os.path.join('losspng','train_loss.csv')
        self.loss_value = []
        # writer = csv.writer(open(tsv, 'w'))
        batch_num = len(train_loader)
        count = 0
        for epoch_idx in tqdm_bar:
            self.model.train()
            acc_loss_train = 0
            count = count+1
            for i,data in tqdm(enumerate(train_loader),total=len(train_loader)):
            # if True:
                # if(i == count):
                    self.optimizer.zero_grad()
                    # GPU
                    graph1, graph2, labels = data[0].to(self.config.device), data[1].to(self.config.device), data[2].to(self.config.device)
                    # data = data.to(self.config.device)
                    #graph1, graph2, labels = data[0], data[1], data[2]  # .to(self.config.device)
                    # print(graph1,labels)
                    # exit(0)
                    loss_train = self.train_epoch_ip(graph1, graph2, labels)
                    loss_train.backward()
                    self.optimizer.step()
                    acc_loss_train += loss_train.detach().cpu().numpy()
                    # break

            if epoch_idx:
            #    writer.writerow([epoch_idx, acc_loss_train])
                self.loss_value.append([epoch_idx, acc_loss_train])
            #每个epoch的损失函数
            # tqdm_bar.set_description('Epoch: {:04d}, loss_train: {:.4f}'.format(epoch_idx, acc_loss_train))
            
            # if epoch_idx % self.config.test_step == 0:
            #     self.evaluate(epoch_idx, train_loader, test_loader,show=0)
    def triplet_train(self, train_loader, mining="semihard"):
        tqdm_bar = tqdm(range(self.config.epochs))
        # with open('train_loss.csv')as f:
        # with open('train_loss.csv', 'w') as f:s
        # tsv = os.path.join('losspng','train_loss.csv')
        # writer = csv.writer(open(tsv, 'w'))
        self.loss_value = []
        batch_num = len(train_loader)
        count = 0
        for epoch_idx in tqdm_bar:
            self.model.train()
            acc_loss_train = 0
            count = count+1
            for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
                self.optimizer.zero_grad()
                # GPU
                # graph1 = data.to(self.config.device)
                # labels = data.label.to(self.config.device)
                graph1 = data
                labels = data.label
                # print(graph1,labels)
                # exit(0)
                # loss_train = self.triplet_train_epoch_ip(graph1, labels)
                g_emb_1, _ = self.model.embed_graph(graph1.x, graph1.edge_index, batch=graph1.batch)
                g_emb_1 = self.model.mlp(g_emb_1)

                distance = distances.CosineSimilarity()
                reducer = reducers.ThresholdReducer(low=0)
                loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)  # margin: 0.2->0.5
                # mining_func = miners.TripletMarginMiner(
                #     margin=0.2, distance=distance, type_of_triplets="semihard"
                # )
                mining_func = miners.TripletMarginMiner(
                    margin=0.2, distance=distance, type_of_triplets=mining
                )

                indices_tuple = mining_func(g_emb_1, labels)
                loss_train = loss_func(g_emb_1, labels, indices_tuple)

                loss_train.backward()
                self.optimizer.step()
                acc_loss_train += loss_train.detach().cpu().numpy()

                # if i % 8 == 0:
                if epoch_idx % 20 == 0:
                    print(
                        "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                            epoch_idx, i, loss_train, mining_func.num_triplets
                        )
                    )

            if epoch_idx:
            #    writer.writerow([epoch_idx, acc_loss_train])
                self.loss_value.append([epoch_idx, acc_loss_train])
            #每个epoch的损失函数
            # tqdm_bar.set_description('Epoch: {:04d}, loss_train: {:.4f}'.format(epoch_idx, acc_loss_train))
            
            # if epoch_idx % self.config.test_step == 0:
            #     self.evaluate(epoch_idx, train_loader, test_loader,show=0)
            self.scheduler.step()
               
    def triplet_train_node_type(self, train_loader, mining="semihard", margin=0.2):
        tqdm_bar = tqdm(range(self.config.epochs))
        # with open('train_loss.csv')as f:
        # with open('train_loss.csv', 'w') as f:s
        # tsv = os.path.join('losspng','train_loss.csv')
        # writer = csv.writer(open(tsv, 'w'))
        self.loss_value = []
        batch_num = len(train_loader)
        count = 0
        for epoch_idx in tqdm_bar:
            self.model.train()
            acc_loss_train = 0
            count = count+1
            for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
                self.optimizer.zero_grad()
                # GPU
                # graph1 = data.to(self.config.device)
                # labels = data.label.to(self.config.device)
                graph1 = data
                labels = data.label
                # print(graph1,labels)
                # exit(0)
                # loss_train = self.triplet_train_epoch_ip(graph1, labels)
                # g_emb_1, _ = self.model.embed_graph(graph1.x, graph1.edge_index, batch=graph1.batch)
                g_emb_1, _ = self.model.embed_graph_node_type(graph1.x, graph1.edge_index, batch=graph1.batch, node_type=graph1.node_type)
                g_emb_1 = self.model.mlp(g_emb_1)

                distance = distances.CosineSimilarity()
                reducer = reducers.ThresholdReducer(low=0)
                loss_func = losses.TripletMarginLoss(margin=margin, distance=distance, reducer=reducer)  # margin: 0.2->0.5
                mining_func = miners.TripletMarginMiner(
                    margin=0.2, distance=distance, type_of_triplets=mining
                )

                indices_tuple = mining_func(g_emb_1, labels)
                loss_train = loss_func(g_emb_1, labels, indices_tuple)

                loss_train.backward()
                self.optimizer.step()
                acc_loss_train += loss_train.detach().cpu().numpy()

                # if i % 8 == 0:
                if epoch_idx % 40 == 0:
                    print(
                        "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                            epoch_idx, i, loss_train, mining_func.num_triplets
                        )
                    )

            if epoch_idx:
            #    writer.writerow([epoch_idx, acc_loss_train])
               self.loss_value.append([epoch_idx, acc_loss_train])
            #每个epoch的损失函数
            # tqdm_bar.set_description('Epoch: {:04d}, loss_train: {:.4f}'.format(epoch_idx, acc_loss_train))
            
            # if epoch_idx % self.config.test_step == 0:
            #     self.evaluate(epoch_idx, train_loader, test_loader,show=0)
            self.scheduler.step()
            
    # @profileit
    def train_epoch_ip(self, graph1, graph2, labels):
        # print(graph1)
        # exit(0)
        g_emb_1, _ = self.model.embed_graph(graph1.x, graph1.edge_index, batch=graph1.batch)
        g_emb_2, _ = self.model.embed_graph(graph2.x, graph2.edge_index, batch=graph2.batch)
        
        # print()
        g_emb_1 = self.model.mlp(g_emb_1)
        g_emb_2 = self.model.mlp(g_emb_2)

        loss_train = self.cos_loss(g_emb_1, g_emb_2, labels)
        # print(loss_train)
        # exit(0)
        return loss_train
    
    # @profileit
    def inference_epoch_ip(self, graph1, graph2):
        g_emb_1, _ = self.model.embed_graph(graph1.x, graph1.edge_index, batch=graph1.batch)
        g_emb_2, _ = self.model.embed_graph(graph2.x, graph2.edge_index, batch=graph2.batch)

        g_emb_1 = self.model.mlp(g_emb_1)
        g_emb_2 = self.model.mlp(g_emb_2)

        similarity = self.cos_sim(g_emb_1, g_emb_2)
        # print(g_emb_1.shape,similarity)
        # exit(0)
        return g_emb_1, g_emb_2, similarity

    def inference_epoch_ip_node_type(self, graph1, graph2):
        # g_emb_1, _ = self.model.embed_graph(graph1.x, graph1.edge_index, batch=graph1.batch)
        # g_emb_2, _ = self.model.embed_graph(graph2.x, graph2.edge_index, batch=graph2.batch)
        g_emb_1, _ = self.model.embed_graph_node_type(graph1.x, graph1.edge_index, batch=graph1.batch, node_type=graph1.node_type)
        g_emb_2, _ = self.model.embed_graph_node_type(graph2.x, graph2.edge_index, batch=graph2.batch, node_type=graph2.node_type)

        g_emb_1 = self.model.mlp(g_emb_1)
        g_emb_2 = self.model.mlp(g_emb_2)

        similarity = self.cos_sim(g_emb_1, g_emb_2)
        # print(g_emb_1.shape,similarity)
        # exit(0)
        return g_emb_1, g_emb_2, similarity
    
    def inference(self, data_loader):
        labels = []
        outputs = []
        total_loss = 0
        with torch.no_grad():
            self.model.eval()
            
            for data in tqdm(data_loader):
                # GPU
                # graph1, graph2, labels_batch = data[0].to(self.config.device), data[1].to(self.config.device), data[2].to(self.config.device)
                graph1, graph2, labels_batch = data[0], data[1], data[2]  # .to(self.config.device)

                g_emb_1, g_emb_2, similarity = self.inference_epoch_ip(graph1, graph2)
                loss = self.cos_loss(g_emb_1, g_emb_2, labels_batch)

                total_loss += loss.detach().cpu().numpy()
                outputs.append(similarity.detach().cpu())
                labels += np.split(labels_batch.detach().cpu().numpy(), len(labels_batch.detach().cpu().numpy()))
            # print(outputs)
            # exit(0)
            outputs = torch.cat(outputs).detach()
            avg_loss = total_loss / (len(data_loader))

            labels_tensor = (torch.LongTensor(labels)> 0).detach() 
            outputs_tensor = torch.FloatTensor(outputs).detach()
            # preds = (outputs > 0.5).detach()
            preds = (outputs > 0.8).detach()  # threshold

        return avg_loss, labels_tensor, outputs_tensor, preds
    
    def inference_node_type(self, data_loader):
        labels = []
        outputs = []
        total_loss = 0
        with torch.no_grad():
            self.model.eval()
            
            for data in tqdm(data_loader):
                # GPU
                graph1, graph2, labels_batch = data[0].to(self.config.device), data[1].to(self.config.device), data[2].to(self.config.device)
                #graph1, graph2, labels_batch = data[0].to("cpu"), data[1].to("cpu"), data[2].to("cpu")

                g_emb_1, g_emb_2, similarity = self.inference_epoch_ip_node_type(graph1, graph2)
                loss = self.cos_loss(g_emb_1, g_emb_2, labels_batch)

                total_loss += loss.detach().cpu().numpy()
                outputs.append(similarity.detach().cpu())
                labels += np.split(labels_batch.detach().cpu().numpy(), len(labels_batch.detach().cpu().numpy()))
            # print(outputs)
            # exit(0)
            outputs = torch.cat(outputs).detach()
            avg_loss = total_loss / (len(data_loader))

            labels_tensor = (torch.LongTensor(labels)> 0).detach() 
            outputs_tensor = torch.FloatTensor(outputs).detach()
            # preds = (outputs > 0.5).detach()
            preds = (outputs > 0.8).detach()  # threshold

        return avg_loss, labels_tensor, outputs_tensor, preds

    def evaluate(self, epoch_idx, train_loader, test_loader,show=1):
        '''评估'''
        # print('hw2vec result')
        train_loss, train_labels, _, train_preds = self.inference(train_loader)
        test_loss, test_labels, _, test_preds = self.inference(test_loader)
        
        if show == 1:
            print("")
            print("Mini Test for Epochs %d:"%epoch_idx)

            self.metric_calc(train_loss, train_labels, train_preds, header="train")
            self.metric_calc(test_loss,  test_labels,  test_preds,  header="test ")

        if self.min_test_loss >= test_loss:
            self.model.save_model(os.path.join("modelparam",f"{self.config.trainer}.cfg"), 
                                  os.path.join("modelparam",f"{self.config.trainer}.pth"))

        # on final evaluate call
        if(epoch_idx==self.config.epochs):
            self.metric_print(self.min_test_loss, **self.metrics, header="best ")

class GraphTrainer(BaseTrainer):
    ''' trainer for graph classification ''' 
    def __init__(self, cfg, class_weights=None):
        super().__init__(cfg)
        self.task = "TJ"
        if class_weights.shape[0] < 2:
            self.loss_func = nn.CrossEntropyLoss()
        else:    
            self.loss_func = nn.CrossEntropyLoss(weight=class_weights.float().to(cfg.device))

    def train(self, data_loader, valid_data_loader):
        tqdm_bar = tqdm(range(self.config.epochs))

        for epoch_idx in tqdm_bar:
            self.model.train()
            acc_loss_train = 0

            for data in data_loader:
                self.optimizer.zero_grad()
                data.to(self.config.device)

                loss_train = self.train_epoch_tj(data)
                loss_train.backward()
                self.optimizer.step()
                acc_loss_train += loss_train.detach().cpu().numpy()

            tqdm_bar.set_description('Epoch: {:04d}, loss_train: {:.4f}'.format(epoch_idx, acc_loss_train))
            
            #每10个epoch评估一次
            if epoch_idx % self.config.test_step == 0:
                self.evaluate(epoch_idx, data_loader, valid_data_loader)

    # @profileit
    def train_epoch_tj(self, data):
        output, _ = self.model.embed_graph(data.x, data.edge_index, data.batch)
        output = self.model.mlp(output)
        output = F.log_softmax(output, dim=1)

        loss_train = self.loss_func(output, data.label)
        return loss_train

    # @profileit
    def inference_epoch_tj(self, data):
        output, attn = self.model.embed_graph(data.x, data.edge_index, data.batch)
        output = self.model.mlp(output)
        output = F.log_softmax(output, dim=1)

        loss = self.loss_func(output, data.label)
        return loss, output, attn
                
    def inference(self, data_loader):
        labels = []
        outputs = []
        node_attns = []
        total_loss = 0
        folder_names = []
        
        with torch.no_grad():
            self.model.eval()
            for i, data in enumerate(data_loader):
                data.to(self.config.device)

                loss, output, attn = self.inference_epoch_tj(data)
                total_loss += loss.detach().cpu().numpy()

                outputs.append(output.cpu())
                
                if 'pool_score' in attn:
                    node_attn = {}
                    node_attn["original_batch"] = data.batch.detach().cpu().numpy().tolist()
                    node_attn["pool_perm"] = attn['pool_perm'].detach().cpu().numpy().tolist()
                    node_attn["pool_batch"] = attn['batch'].detach().cpu().numpy().tolist()
                    node_attn["pool_score"] = attn['pool_score'].detach().cpu().numpy().tolist()
                    node_attns.append(node_attn)

                labels += np.split(data.label.cpu().numpy(), len(data.label.cpu().numpy()))

            outputs = torch.cat(outputs).reshape(-1,2).detach()
            avg_loss = total_loss / (len(data_loader))

            labels_tensor = torch.LongTensor(labels).detach()
            outputs_tensor = torch.FloatTensor(outputs).detach()
            preds = outputs_tensor.max(1)[1].type_as(labels_tensor).detach()

        return avg_loss, labels_tensor, outputs_tensor, preds, node_attns

    def evaluate(self, epoch_idx, data_loader, valid_data_loader):
        train_loss, train_labels, _, train_preds, train_node_attns = self.inference(data_loader)
        test_loss, test_labels, _, test_preds, test_node_attns = self.inference(valid_data_loader)

        print("")
        print("Mini Test for Epochs %d:"%epoch_idx)

        self.metric_calc(train_loss, train_labels, train_preds, header="train")
        self.metric_calc(test_loss,  test_labels,  test_preds,  header="test ")

        if self.min_test_loss >= test_loss:
            self.model.save_model("modeltj.cfg", "modeltj.pth")

            #TODO: store the attn_weights right here. 

        # on final evaluate call
        if(epoch_idx==self.config.epochs):
            self.metric_print(self.min_test_loss, **self.metrics, header="best ")


class Evaluator(BaseTrainer):
    def __init__(self, cfg, task):
        super().__init__(cfg)
        self.task = task