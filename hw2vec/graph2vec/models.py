#!/usr/bin/env python
#title           :models.py
#description     :This file includes the models of hw2vec.
#author          :Shih-Yuan Yu
#date            :2021/03/05
#version         :0.2
#notes           :
#python_version  :3.6
#==============================================================================
import json
import copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch.nn import Linear, ReLU
from torch_geometric.nn import GCNConv, GINConv, SAGPooling, TopKPooling,norm
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

class GRAPH_CONV(nn.Module):
    def __init__(self, type, in_channels, out_channels):
        super(GRAPH_CONV, self).__init__()
        self.type = type
        self.in_channels = in_channels
        self.out_channels = out_channels
        if type == "gcn":
            self.graph_conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.graph_conv(x, edge_index)

class GRAPH_POOL(nn.Module):
    def __init__(self, type, in_channels, poolratio):
        super(GRAPH_POOL, self).__init__()
        self.type = type
        self.in_channels = in_channels
        self.poolratio = poolratio
        if self.type == "sagpool":
            self.graph_pool = SAGPooling(in_channels, ratio=poolratio)
        elif self.type == "topkpool":
            self.graph_pool = TopKPooling(in_channels, ratio=poolratio)
    
    def forward(self, x, edge_index, batch):
        return self.graph_pool(x, edge_index, batch=batch)

class GRAPH_READOUT(nn.Module):
    def __init__(self, type):
        super(GRAPH_READOUT, self).__init__()
        self.type = type
    
    def forward(self, x, batch):
        if self.type == "max":
            return global_max_pool(x, batch)
        elif self.type == "mean":
            return global_mean_pool(x, batch)
        elif self.type == "add":
            return global_add_pool(x, batch)


class GRAPH2VEC(nn.Module):
    
    ''' 
        该类即将处理好的图用来训练的模块 \n
        For users who want to develop their own network architecture, 
        you may use this graph2vec class as template and implement your architecture.
    '''

    def __init__(self, config):
        super(GRAPH2VEC, self).__init__()
        self.config = config
        self.cfg    = config
        # 10 node types in a gate-level netlist: input, and, nand, or, nor,
        # not, buf, xor, xnor, output.
        if getattr(config, "num_node_type", None) is None:
            config.num_node_type = 10
        self.bn     = norm.BatchNorm(config.hidden*3*2).to(config.device)
    def save_model(self, model_config_path, model_weight_path):
        Path(model_config_path).parent.mkdir(parents=True, exist_ok=True)
        Path(model_weight_path).parent.mkdir(parents=True, exist_ok=True)
        model_configurations = {}

        convs = [] 
        for layer in self.layers:
            convs.append((layer.type, layer.in_channels, layer.out_channels))
        model_configurations['convs'] = convs

        model_configurations['pool'] = (self.pool1.type, self.pool1.in_channels, self.pool1.poolratio)
        model_configurations['readout'] = self.graph_readout.type
        model_configurations['fc'] = (self.fc.in_features, self.fc.out_features)
        with open(model_config_path, 'w') as f:
            json.dump(model_configurations, f)
        torch.save(self.state_dict(), model_weight_path)

    def load_model(self, model_config_path, model_weight_path):
        with open(model_config_path) as f:
            
            model_configuration = json.load(f)
            print(model_configuration)
        
        convs = [] 
        for setting in model_configuration['convs']:
            graph_conv_type, in_channels, out_channels = setting
            convs.append(GRAPH_CONV(graph_conv_type, int(in_channels), int(out_channels)))
        self.set_graph_conv(convs)

        pool_type, pool_in_channels, pool_ratio = model_configuration['pool']
        self.set_graph_pool(GRAPH_POOL(pool_type, pool_in_channels, pool_ratio))

        self.set_graph_readout(GRAPH_READOUT(model_configuration['readout']))
        fc_in_channel, fc_out_channel = model_configuration['fc']
        self.set_output_layer(nn.Linear(fc_in_channel, fc_out_channel))
        print("debug",model_weight_path)
        # exit(0)
        self.load_state_dict(torch.load(model_weight_path))
        print("load ok")
        
    def set_graph_conv(self, convs):
        self.layers = []
        
        for conv in convs:
            conv.to(self.config.device)
            self.layers.append(conv)
        self.layers = nn.ModuleList(self.layers)

    def set_graph_conv_node_type(self, convs):
        """Build one independent convolution per node type, for every layer.

        Node-independent message passing needs a separate weight set for each
        node type, so every layer is expanded into ``num_node_type`` distinct
        modules. The layers are kept in a ``nn.ModuleList`` so that they are
        registered on the module and therefore reachable from
        ``parameters()``, ``to(device)`` and ``state_dict()``.
        """
        num_node_type = self.config.num_node_type
        layers = []

        for conv in convs:
            conv.to(self.config.device)
            temp = [copy.deepcopy(conv) for _ in range(num_node_type)]
            layers.append(nn.ModuleList(temp))

        self.layers = nn.ModuleList(layers)

    def set_graph_pool(self, pool_layer):
        self.pool1 = pool_layer.to(self.config.device)
            
    def set_graph_readout(self, typeofreadout):
        self.graph_readout = typeofreadout

    def set_output_layer(self, layer):
        self.fc = layer.to(self.config.device)
    
    def set_node_output_layer(self, layer):
        self.node_fc = layer.to(self.config.device)
    
    def set_attention_output_layer(self, layer):
        self.att_liner= layer.to(self.config.device)
        
    def new_set_output_layer(self, liners):
        self.ne_fc = []
        for layer in liners:
            layer.to(self.config.device)
            self.ne_fc.append(layer)
        self.ne_fc= nn.ModuleList(self.ne_fc)
    
    def embed_graph(self, x, edge_index, batch):
        attn_weights = dict()
        # print(len(x))
        # x = F.one_hot(x, num_classes=self.config.num_feature_dim).float()  # self-made one-hot
        # print(x[0],x[1])
        # exit(0)
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x, edge_index)), p=self.config.dropout, training=self.training)
            # print(x.shape)
        # exit(0)
        # pool
        # x, edge_index, _, batch, attn_weights['pool_perm'], attn_weights['pool_score'] = \
        #     self.pool1(x, edge_index, batch=batch)
        # print(x.shape)
        x = self.graph_readout(x, batch)
        # print(x.shape)
        # exit(0)
        attn_weights['batch'] = batch
        return x, attn_weights
    
    def embed_graph_node_type(self, x, edge_index, batch, node_type):
        attn_weights = dict()
        # print(len(x))
        # x = F.one_hot(x, num_classes=self.config.num_feature_dim).float()  # self-made one-hot
        # print(x[0],x[1])
        # exit(0)
        for layer in self.layers:
            temp = []
            for i in range(self.config.num_node_type):
                # mask = (node_type == i)
                zz = layer[i](x, edge_index)
                temp.append(zz)
            # tempx = torch.zeros(x.shape[0], temp[0].shape[1]).to(self.config.device)
            tempx = torch.zeros(x.shape[0], temp[0].shape[1], device=self.config.device)
            for i in range(self.config.num_node_type):
                mask = (node_type == i)
                # tempx = torch.zeros(x.shape[0], temp[0].shape[1]).to(self.config.device)  # NCRZ
                tempx[mask] = temp[i][mask]
            x = F.dropout(F.relu(tempx), p=self.config.dropout, training=self.training)
            # print(x.shape)
        # exit(0)
        # x, edge_index, _, batch, attn_weights['pool_perm'], attn_weights['pool_score'] = \
            # self.pool1(x, edge_index, batch=batch)
        # print(x.shape)
        x = self.graph_readout(x, batch)
        # print(x.shape)
        # exit(0)
        attn_weights['batch'] = batch
        return x, attn_weights
    
    def cgs_embed_graph(self, x, edge_index, batch):
        attn_weights = dict()
        # print(len(x))
        x = F.one_hot(x, num_classes=self.config.num_feature_dim).float()
        # print(x[0],x[1])
        # exit(0)
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x, edge_index)), p=self.config.dropout, training=self.training)
        #     print(x.shape)
        # exit(0)
        # node_encode = x.detach()
        x, edge_index, _, batch, attn_weights['pool_perm'], attn_weights['pool_score'] = \
            self.pool1(x, edge_index, batch=batch)
        node_encode = x.detach()  # modified
        x = self.graph_readout(x, batch)
   
        attn_weights['batch'] = batch
        return node_encode,x
    
    def aggrate(self,x,edge_index,batch):
        attn_weights = dict()
        # x, edge_index, _, batch, attn_weights['pool_perm'], attn_weights['pool_score'] = \
        # self.pool1(x, edge_index, batch=batch)
        c_l = self.graph_readout(x, batch) #[batch,hidden]
        
        c_l = self.att_mlp(c_l)#非线性变换  #[batch,hidden]
        c_l = torch.repeat_interleave(c_l,torch.bincount(batch),dim=0)#[node_num,hidden]
        
        att_matrax = torch.matmul(x,c_l.t())#[node_num,node_num]
        
        x = F.sigmoid(torch.matmul(att_matrax,x))#[node_num,hidden]
        x = scatter(x,batch)#[batch,hidden]
        # print(x.shape)
        # exit(0)
        return x
    
    def new_embed_graph(self, x, edge_index, batch):
        attn_weights = dict()
        x = F.one_hot(x, num_classes=self.config.num_feature_dim).float()
        # print('debug',x.shape)
        # exit(0)
        results = []
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x, edge_index)), p=self.config.dropout, training=self.training)
            # print(x.shape)
            # exit(0)
            results.append(self.aggrate(x.detach(),edge_index.detach(),batch.detach()))
            # print(results[0].shape)
            # exit(0)
        contract_y = torch.cat((results[0],results[1],results[2]),dim=-1).float()
        # print(contract_y.shape)
        # exit(0)
        # x, edge_index, _, batch, attn_weights['pool_perm'], attn_weights['pool_score'] = \
        #     self.pool1(x, edge_index, batch=batch)
        # x = self.graph_readout(x, batch)
            

        attn_weights['batch'] = batch
        return contract_y, attn_weights    
    
    def embed_node(self, x, edge_index):
        x = F.one_hot(x, num_classes=self.config.num_feature_dim).float()
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x, edge_index)), p=self.config.dropout, training=self.training)
        return x

    def mlp(self, x):
        return self.fc(x)
    
    def node_mlp(self, x):
        return self.fc(x)
     
    def new_mlp(self, x):
        num = len(self.ne_fc)
        for i in range(num-1):
            x = F.relu(self.ne_fc[i](x))
        return F.relu(self.ne_fc[num-1](x))
    
    def att_mlp(self, x):
        return F.sigmoid(self.att_liner(x))


