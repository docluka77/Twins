#!/usr/bin/env python
#title           :hw2graph.py
#description     :This file includes the neccessary functions in hw2graph.
#author          :Rozhin
#date            :2021/03/05
#version         :0.2
#notes           :
#python_version  :3.6
#==============================================================================
from __future__ import absolute_import
from __future__ import print_function

import pyverilog.utils.util as util
import networkx as nx
import itertools
import pyverilog, pydot, json, os, sys, pickle
sys.path.append(os.path.dirname(sys.path[0]))

from json import dumps
from collections import defaultdict
from pyverilog.dataflow.optimizer import VerilogDataflowOptimizer
from pyverilog.dataflow.graphgen import VerilogGraphGenerator
from pyverilog.controlflow.controlflow_analyzer import VerilogControlflowAnalyzer
from pyverilog.vparser.parser import parse
from sklearn.model_selection import train_test_split
from glob import glob
from hw2vec.graph2vec.trainers import *
from hw2vec.utilities import *
from torch_geometric.utils.convert import from_networkx
from time import time

class DataProcessor:
    '''图处理类'''
    def __init__(self, cfg):
        self.cfg = cfg
        self.graph_data = []
        self.graph_pair_data = []
         
        '''AST 的值'''
        self.global_type2idx_AST_list = ['names','always','none','senslist','sens','identifier','nonblockingsubstitution',
                                         'lvalue','rvalue','intconst','pointer','ifstatement','pure numeric','assign','cond',
                                         'unot','plus','land','reg','partselect','eq','lessthan','greaterthan','decl','wire',
                                         'width','output','input','moduledef','portarg','instancelist','source','description',
                                         'port','portlist','ulnot','instance','or','and','lor','block','xor','ioport',
                                         'blockingsubstitution','minus','times','casestatement','case','parameter','sll','srl',
                                         'sra','divide','systemcall','singlestatement','stringconst','noteq','concat','repeat',
                                         'integer','xnor','dimensions','length','lconcat','uminus','greatereq','initial','uor',
                                         'casexstatement','forstatement','localparam','eventstatement','mod','delaystatement',
                                         'floatconst','task','paramarg', 'paramlist', 'inout']
        self.global_type2idx_AST = {v:k for k, v in enumerate(self.global_type2idx_AST_list)}
        
        '''DFG的值'''
        self.global_type2idx_DFG_list = ['concat','input','unand','unor','uxor','signal','uand','ulnot','uxnor','numeric','partselect',
                                         'and','unot','branch','or','uor','output','plus','eq','minus','xor','lor','noteq','land',
                                         'greatereq','greaterthan','sll','lessthan','times','srl','pointer','mod','divide','sra','sla',
                                         'xnor', 'lesseq']

        self.global_type2idx_DFG = {v:k for k, v in enumerate(self.global_type2idx_DFG_list)}

    def process(self, nx_graph):
        '''处理输入的图'''
        self.normalize(nx_graph)
        data = from_networkx(nx_graph)
        data.hw_name = nx_graph.name
        data.hw_type = nx_graph.type
        self.graph_data.append(data)

    def normalize(self, nx_graph):
        ''' 
            用数值代替标签 \n
            Normalization is a step to replace the 
            label of a node to a value -> replace all the variable name 
            and numeric values with a high-level type.
        '''
        if self.cfg.graph_type == 'DFG': # normalize for DFG
            in_degrees = [val for (node, val) in nx_graph.in_degree()]
            out_degrees = [val for (node, val) in nx_graph.out_degree()]
            for idx, node in enumerate(nx_graph.nodes(data=True)):
                node_name = node[0]
                if '_graphrename' in node_name:
                    node_name = node_name[:node_name.index('_graphrename')]
                if '\'d' in node_name or '\'b' in node_name or '\'o' in node_name or '\'h' in node_name: #转换进制的数
                    type_of_node = "numeric"
                elif in_degrees[idx] == 0:   #入度为0输出
                    type_of_node = "output"
                elif out_degrees[idx] == 0:  #出度为0输入
                    type_of_node = "input"
                elif '.' in node_name or '_' in node_name: #信号
                    type_of_node = "signal"
                else:
                    type_of_node = node_name.lower()

                if type_of_node not in self.global_type2idx_DFG:
                    print("----------"+type_of_node+"-------------")
                    raise Exception("The operation is not in the global_type2idx_DFG table, please report the error to " +   
                                "https://github.com/louisccc/hw2vec/issues")
                
                node[1]['x'] = self.global_type2idx_DFG[type_of_node]
            self.num_node_labels = len(self.global_type2idx_DFG)
            self.cfg.num_feature_dim = self.num_node_labels
        
        elif self.cfg.graph_type == "AST": # normalize for AST
            out_degrees = [val for (node, val) in nx_graph.out_degree()]
            for idx, node in enumerate(nx_graph.nodes(data=True)):
                label = node[1]['label']
                if out_degrees[idx] == 0 and not isInt(label):
                    type_of_node = "names"
                elif isInt(label):
                    type_of_node = "pure numeric"
                else:
                    type_of_node = label.lower()
                
                if type_of_node not in self.global_type2idx_AST:
                    print("----------"+type_of_node+"-------------")
                    raise Exception("The operation is not in the global_type2idx_AST table, please report the error to " +   
                                "https://github.com/louisccc/hw2vec/issues")
                
                node[1]['x'] = self.global_type2idx_AST[type_of_node]
            self.num_node_labels = len(self.global_type2idx_AST)
            self.cfg.num_feature_dim = self.num_node_labels

    def generate_pairs(self):
        '''生成所有图形的两两组合'''
        self.graph_pair_data = []
        # print('len of graph',len(self.graph_data))
        for idx_graph_a, idx_graph_b in itertools.combinations(range(len(self.graph_data)), 2):
            self.graph_pair_data.append((self.graph_data[idx_graph_a], self.graph_data[idx_graph_b]))
        #TODO: different sampling technique?
    
    def get_graphs(self):
        return self.graph_data
    
    def get_pairs(self):
        '''返回所有图形的两两组合'''
        if len(self.graph_pair_data) == 0:
            raise Exception("you have to genearte pairs first!")
        return self.graph_pair_data

    def split_dataset(self, ratio, seed, dataset):
        '''分割数据集  参数是分割比例、种子、数据集'''
        sim_diff_label = []

        for data in dataset:
            if type(data) == tuple:  #如果是元组
                if data[2] == 1:
                    sim_diff_label.append(1)   #相似标签
                else:
                    sim_diff_label.append(-1)  #不相似标签
            else:
                if data.label == 1:
                    sim_diff_label.append(1)
                else:
                    sim_diff_label.append(-1)

        return train_test_split(dataset, train_size = int(len(dataset) * ratio), shuffle = True, stratify=sim_diff_label, random_state=seed)

    def get_class_weights(self, train_graphs):
        training_labels = [data.label for data in train_graphs]
        class_weights = torch.from_numpy(compute_class_weight(class_weight= 'balanced',classes= np.unique(training_labels),y= training_labels))

        return class_weights

    def cache_graph_data(self, data_pkl_path):
        with open(data_pkl_path, 'wb+') as f:
            pickle.dump(self.graph_data, f)
    
    def new_cache_graph_data(self, data_pkl_path,graph):
        with open(data_pkl_path, 'wb+') as f:
            pickle.dump(graph, f)
        
    def read_graph_data_from_cache(self, data_pkl_path):
        with open(data_pkl_path, 'rb') as f:
            # GPU
            # self.graph_data = pickle.load(f)
            self.graph_data = torch.load(f)

        if self.cfg.graph_type == 'DFG':
            self.num_node_labels = len(self.global_type2idx_DFG)
            self.cfg.num_feature_dim = self.num_node_labels
        elif self.cfg.graph_type == 'AST':
            self.num_node_labels = len(self.global_type2idx_AST)
            self.cfg.num_feature_dim = self.num_node_labels
        #TODO: creating pairs or blah blah blah.


class DFGGenerator:
    def __init__(self):
        pass

    def process(self, verilog_file):
        dataflow_analyzer = VerilogDataflowAnalyzer(verilog_file, "top")
        dataflow_analyzer.generate()
        binddict = dataflow_analyzer.getBinddict()
        terms = dataflow_analyzer.getTerms()
        
        dataflow_optimizer = VerilogDataflowOptimizer(terms, binddict)
        dataflow_optimizer.resolveConstant()
        resolved_terms = dataflow_optimizer.getResolvedTerms()
        resolved_binddict = dataflow_optimizer.getResolvedBinddict()
        constlist = dataflow_optimizer.getConstlist()
        dfg_graph_generator = VerilogGraphGenerator("top", terms, binddict, resolved_terms, 
                                                    resolved_binddict, constlist, './seperate_modules.pdf')

        # binddict with string keys
        signals = [str(bind) for bind in dfg_graph_generator.binddict]

        for num, signal in enumerate(sorted(signals, key=str.casefold), start=1):
            dfg_graph_generator.generate(signal, walk=False)
            # print(f'\rProgress : {num} / {len(signals)}', end='', flush=True)

        label_to_node = dict()
        for node in dfg_graph_generator.graph.nodes():
            if dfg_graph_generator.graph.in_degree(node) == 0:
                label = node.attr['label'] if node.attr['label'] != '\\N' else str(node)
                label_to_node[label] = node
        
        num_nodes = len(dfg_graph_generator.graph.nodes())
        for num, node in enumerate(dfg_graph_generator.graph.nodes(), start=1):
            label = node.attr['label'] if node.attr['label'] != '\\N' else str(node)
            if '_' in label and label.replace('_', '.') in label_to_node:
                parents = dfg_graph_generator.graph.predecessors(node)
                dfg_graph_generator.graph.delete_node(node)
                for parent in parents:
                    dfg_graph_generator.graph.add_edge(parent, label_to_node[label.replace('_', '.')])

        
        nx_graph = nx.DiGraph()
        
        for node in dfg_graph_generator.graph.nodes():
            node_name = node.name
            if '_graphrename' in node.name:
                node_name = node.name[:node.name.index('_graphrename')]
            if '.' in node_name: 
                type_of_node = node_name.split('.')[-1]
            elif '_' in node_name:
                type_of_node = node_name.split('_')[-1]
            else:
                type_of_node = node_name.lower()
            nx_graph.add_node(node.name, label=type_of_node) 
            for child in dfg_graph_generator.graph.successors(node):
                # edgeLabel = dfg_graph_generator.graph.get_edge(node, child).attr['label']
                nx_graph.add_edge(node.name, child.name)

        return nx_graph


class ASTGenerator:
    def __init__(self):
        self.DICTIONARY_GEN = \
            ["Source","Description","Ioport","Decl","Lvalue"]
        self.ARRAY_GEN = \
            ["ModuleDef","Paramlist","Portlist","Input","Width","Reg","Wire","Rvalue","ParseSelect",
             "Uplus","Uminus","Ulnot","Unot","Uand","Unand","Uor","Unor","Uxnor","Power","Times","Divide","Mod","Plus",
             "Minus","Sll","Srl","Sla","Sra","LessThan","GreaterThan","LessEq","GreaterEq","Eq","Eql","NotEq","Eql","NotEql",
             "And","Xor","Xnor","Or","Land","Lor","Cond","Assign","Always","AlwaysFF","AlwaysComb","AlwaysLatch",
             "SensList","Sens","Substitution","BlockingSubstitution","NonblockingSubstitution","IfStatement","Block",
             "Initial","Plus","Output","Partselect","Port","InstanceList","Instance","PortArg","Pointer","Concat", "Parameter", 
             "SystemCall", "CaseStatement", "Case", "Function", "CasezStatement", "FunctionCall", "Dimensions", "Length", 
             "LConcat", "Concat", "SingleStatement", "Repeat", "Integer", "CasexStatement", "ForStatement", "Localparam",
             "EventStatement", "DelayStatement", "Task", "ParamArg", "Inout"]
        self.CONST_DICTIONARY_GEN = \
            ["IntConst","FloatConst","StringConst","Identifier"]
        self.count = 0

    def process(self, verilog_file):
        #when generating AST, determines which substructure (dictionary/array) to generate
        #before converting the json-like structure into actual json
        
        self.ast, _ = parse([verilog_file], debug=False)
        ast_dict = self._generate_ast_dict(self.ast)

        nx_graph = nx.DiGraph()
        for key in ast_dict.keys():
            self._add_node(nx_graph, 'None', key, ast_dict[key])

        return nx_graph

    #generates nested dictionary for conversion to json (AST helper)
    def _generate_ast_dict(self, ast_node):
        class_name = ast_node.__class__.__name__
        structure = {}
        #based on the token class_name, determine the value type of class_name
        if class_name in self.ARRAY_GEN:
            structure[class_name] = [getattr(ast_node, n) for n in ast_node.attr_names] if ast_node.attr_names else []
            for c in ast_node.children():
                structure[class_name].append(self._generate_ast_dict(c))
        elif class_name in self.DICTIONARY_GEN:
            structure[class_name] = self._generate_ast_dict(ast_node.children()[0])
        elif class_name in self.CONST_DICTIONARY_GEN:
            structure = {}
            structure[class_name] = getattr(ast_node,ast_node.attr_names[0])
            return structure
        else:
            raise Exception(f"Error. Token name {class_name} is invalid or has not yet been supported")
        return structure
    
    def _add_node(self, graph, parent, child, cur_dict):
        index = self.count
        graph.add_nodes_from([(index, {"label": str(child)})])
        if parent != 'None':
            graph.add_edge(parent, index)
        self.count = self.count + 1
        if type(cur_dict) == dict:
            for key in cur_dict.keys():
                self._add_node(graph, index, key, cur_dict[key])
        elif type(cur_dict) == list:
            for ele in cur_dict:
                if type(ele) == dict:
                    self._add_node(graph, index, 'None', ele)
                elif ele is not None:
                    graph.add_nodes_from([(self.count, {"label": str(ele)})])
                    graph.add_edge(index, self.count)
                    self.count = self.count + 1
        else:
            graph.add_nodes_from([(self.count, {"label": str(cur_dict)})])
            graph.add_edge(index, self.count)
            self.count = self.count + 1

class CFGGenerator:
    def __init__(self):
        pass

    def process(self, verilog_file): 
        '''处理CFG(control flow graph)'''
        fsm_vars = tuple(['fsm', 'state', 'count', 'cnt', 'step', 'mode'])   
        dataflow_analyzer = VerilogDataflowAnalyzer(self.verilog_file, "top")
        dataflow_analyzer.generate()
        binddict = dataflow_analyzer.getBinddict()
        terms = dataflow_analyzer.getTerms()
        
        #优化数据流
        dataflow_optimizer = VerilogDataflowOptimizer(terms, binddict)
        #常量解析
        dataflow_optimizer.resolveConstant()
        #常量解析结果
        resolved_terms = dataflow_optimizer.getResolvedTerms()
        #输入输出的字典关系
        resolved_binddict = dataflow_optimizer.getResolvedBinddict()
        #常量列表
        constlist = dataflow_optimizer.getConstlist()
        #获取有限状态自动机
        cfg_graph_generator = VerilogControlflowAnalyzer("top", terms, binddict, resolved_terms, resolved_binddict, constlist, fsm_vars)
        fsms = cfg_graph_generator.getFiniteStateMachines()

        print("VIEWING FSM's")
        print("LENGTH OF FSM: ", len(fsms))
        for signame, fsm in fsms.items():
            print('# SIGNAL NAME: %s' % signame)
            print('# DELAY CNT: %d' % fsm.delaycnt)
            fsm.view()
            fsm.tograph(filename=util.toFlatname(signame) + '.png', nolabel=False)

        graph = pydot.graph_from_dot_file("./file.dot")[0]
        #节点
        nodes = [node.get_name() for node in graph.get_node_list()]
        #根节点
        root_nodes = [node.get_name() for node in graph.get_node_list() if node.obj_dict['parent_graph'] == None]
        #边
        edges = [[edge.get_source(),edge.get_destination(),edge.obj_dict['attributes']['label']] for edge in graph.get_edge_list()]
        topModule = defaultdict([]) 

        for edge in edges:
            if edge[2] == 'None':
                topModule[edge[0]].append("")
            else:
                topModule[edge[0]].append(edge[2])
        
        if (output=='roots'):
            print(f'Saving all {len(root_nodes)} nodes as json')
            with open('./root_nodes.json', 'w') as f:
                f.write(dumps(root_nodes, indent=4))
            print('List of root nodes saved in root_nodes.json.\n')
            f.close()
        
        elif (output=='nodes'):
            print(f'Saving all {len(nodes)} nodes as json')
            with open('./all_nodes.json', 'w') as f:
                f.write(dumps(nodes, indent=4))
            print('List of nodes saved in all_nodes.json.\n')
            f.close()
            
        elif (output=='edges'):
            print(f'Saving all {len(edges)} edges as json')
            with open('./all_edges.json', 'w') as f:
                f.write(dumps(edges, indent=4))
            print('List of edges saved in all_edges.json.\n')
            f.close()
            
        elif (output=='graph'):
            print(f'Saving cfg graph dictionary as a json')
            with open('./topModule.json', 'w') as f:
                f.write(dumps(topModule, indent=4))
            print('Saving cfg graph dictionary as a json.\n')
            f.close()
            print('The graph is saved as topModule.json.\n')
        return None


class HW2GRAPH:
    
    '''
        The main class of hw2graph consists of two components:
        1. preprocess (flatten, remove comment, ...)
        2. processs (call backend graph geenration and acquire the nx graph instances.)

        Currently HW2GRAPH can process Verilog files in RTL (Register Transfer Level) and GLN (Gate-Level Netlist).
    ''' 

    def __init__(self, cfg):
        self.cfg = cfg

    def find_hw_project_folders(self):        
        projects = set()
        
        # ** 不只查找当前目录还查找子目录
        for verilog_path in glob("%s/**/*.v" % str(self.cfg.raw_dataset_path), recursive=True):
            #添加文件夹路径
            folder_name = Path(verilog_path).parent
            projects.add(folder_name)
        # print(projects)
        return list(projects)

    def flatten(self, input_path, flattened_hw_path):
        '''将所有.v中的写到一个文件，会检查是否有top_moudle.v'''
        flatten_content = ""
        all_containing_files = [Path(x).name for x in glob(fr'{input_path}/*.v', recursive=True)]
        if "topModule.v" in all_containing_files:
            return
        for verilog_file in glob(fr'{input_path}/*.v'):
            with open(verilog_file, "r") as infile:
                flatten_content += infile.read()     
        with open(flattened_hw_path, "w") as outfile:
            outfile.write(flatten_content)       

    def remove_comments(self, hw_path):
        '''去除注释，找到 // 然后把之后的去除'''
        
        with open(hw_path,'r') as file_in:
            lines = file_in.read().split("\n")
    
        #TODO; right now this part is a rule-based method, we will consider using AST to remove comments in the future.
        with open(hw_path, "w") as file_out:
            for line in lines:
                idx = line.find('//')
                if idx == 0:
                    continue
                elif idx == -1:
                    file_out.write(line+'\n')
                else:
                    file_out.write(line[:idx]+'\n')
        
    def remove_underscores(self, hw_path):
        '''去除下划线'''
        with open(hw_path, 'r') as file_in:
            lines = file_in.read().replace('_', '')

        with open(hw_path, "w") as file_out:
            file_out.write(lines)

    
    def rename_topModule(self, hw_path):
        '''对module计数并得到顶层模块'''
        #TODO; right now this part is a rule-based method, 
        # we will consider using AST to parse the flattened code in the future.
        with open(hw_path,'r') as file_in:
            lines = file_in.read().split("\n")
    
        modules_dic={}
        '''定义module和引用moudle'''
        for line in lines:
            words = line.split()
            for word_idx, word in enumerate(words):
                if word == 'module':
                    module_name = words[word_idx+1]
                    if '(' in module_name:
                        idx = module_name.find('(')
                        module_name = module_name[:idx]
                        modules_dic[module_name]= 1

                    else:
                        modules_dic[module_name]= 0
        
        '''module引用次数加一'''
        for line in lines:
            words = line.split()
            for word in words:
                if word in modules_dic.keys():
                    modules_dic[word] += 1
        
        '''只用了一次就是顶层module'''
        for m in modules_dic:
            if modules_dic[m] == 1:
                top_module = m
                break
        '''写回，将顶层moudle替换为top'''
        with open(hw_path, "w") as file_out:
            for line in lines:
                file_out.write(line.replace(top_module, 'top')+'\n')        

    def code2graph(self, verilog_dir, profile=True):
        if profile:
            start = time()

        code_path = self.preprocess(verilog_dir)
        hw_graph = self.process(str(code_path))
        
        if profile:
            end = time()
            print(str(code_path), ",", len(hw_graph.nodes), ",", len(hw_graph.edges), ",", end-start)
        return hw_graph
        
    def preprocess(self, verilog_dir):
        '''处理所有 .v文件并得到顶层模块名字'''
        # flattened_hw_path = verilog_dir / "topModule.v"  # ???
        flattened_hw_path = os.path.join(verilog_dir, "topModule.v")
        all_verilog_files = [Path(x).name for x in glob("%s/*.v"%str(verilog_dir))]

        if "topModule.v" not in all_verilog_files:
            self.flatten(verilog_dir, flattened_hw_path)
            self.remove_comments(flattened_hw_path)
            self.remove_underscores(flattened_hw_path)
            self.rename_topModule(flattened_hw_path)
        print(flattened_hw_path)
        return flattened_hw_path    

    # @profilegraph
    def process(self, hw_path):
        '''传入原始处理后硬件语言并处理'''
        if self.cfg.graph_type == "CFG":
            
            generator = CFGGenerator()
            return_obj = generator.process(hw_path)
            nx_graph = None
        
        elif self.cfg.graph_type == "AST":
            generator = ASTGenerator()
            nx_graph = generator.process(hw_path)

        elif self.cfg.graph_type == "DFG":
            generator = DFGGenerator()
            nx_graph = generator.process(hw_path)
        
        else:
            pass
            
        for file in ['file.dot','parser.out','parsetab.py','top_state.png']:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass

        if nx_graph != None:
            # NOTE: this is creating a limitation of how users should form their dataset. (TODO: we have to provide a tutorial in readme.)
            if sys.platform.startswith('win'):
               nx_graph.name = hw_path.split("\\")[-2]
               nx_graph.type = hw_path.split("\\")[-3]
            elif sys.platform.startswith('linux'):
               nx_graph.name = hw_path.split("/")[-2]
               nx_graph.type = hw_path.split("/")[-3]

            
        return nx_graph