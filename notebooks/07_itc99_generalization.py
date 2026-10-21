import os, sys, itertools
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from hw2vec.config import Config
from hw2vec.hw2graph import *
from tqdm import tqdm
from pytorch_metric_learning import distances, losses, miners, reducers
# dir = ['--yaml_path', '../config/example_gnn4ip.yaml', \
#     #    '--raw_dataset_path', '../../bigcreate/newdatas', \
#     #    '--data_pkl_path', 'pkl/dfg_ip_rt291.pkl', \
#         '--graph_type', 'DFG']
dir = ['--yaml_path', '../config/example_gnn4ip.yaml']
# cfg = Config(sys.argv[1:])
cfg = Config(dir)
cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg.num_feature_dim = 10
cfg.batch_size = 16
cfg.hidden1 = 64
cfg.hidden2 = 128
cfg.embed_dim = 256
# cfg.epochs = 200
# cfg.epochs = 100
cfg.epochs = 300
# cfg.epochs = 400
# cfg.learning_rate = 0.001
cfg.learning_rate = 0.005
# cfg.learning_rate = 0.05
cfg.dropout = 0.1
# dataset = torch.load("Reduced-DFG-with_big.pt")
# dataset = torch.load("Reduced-DFG.pt")
dataset = torch.load("NetlistGraph-node_type-ITC99.pt")
sim_diff_label = []
for data in dataset:
    sim_diff_label.append(data.label)
    #trainset, testset = train_test_split(dataset, \
#                                     train_size = int(len(dataset) * cfg.ratio), \
#                                     shuffle=True, \
#                                     #  stratify=None, \
#                                     stratify=sim_diff_label, \
#                                     random_state=cfg.seed)
trainset, testset = train_test_split(dataset, 
                                     train_size=int(len(dataset) * cfg.ratio), 
                                     shuffle=True, 
                                     random_state=cfg.seed)
train_loader = DataLoader(dataset, shuffle=True, batch_size=cfg.batch_size)
test_pairs = []
for idx_graph_a, idx_graph_b in itertools.combinations(range(len(testset)), 2):  # testset
    test_pairs.append((testset[idx_graph_a], testset[idx_graph_b]))

SIMILAR = torch.tensor(1, device=cfg.device)
DISSIMILAR = torch.tensor(-1, device=cfg.device)
for pair_idx, pair in enumerate(test_pairs):
    # print(type(all_pairs))
    graph_a, graph_b = pair
    # print(all_pairs[pair_idx], graph_a.hw_type, graph_b.hw_type)
    if graph_a.label == graph_b.label:
        test_pairs[pair_idx] += (SIMILAR,)
    else:
        test_pairs[pair_idx] += (DISSIMILAR,)
        
test_pairs[0]
test_loader = DataLoader(test_pairs, batch_size=cfg.batch_size)
model = GRAPH2VEC(cfg)
convs = [
    GRAPH_CONV("gcn", cfg.num_feature_dim, cfg.hidden1),  # 前者是标签数量，后者是隐藏层
    # GRAPH_CONV("gcn", cfg.hidden, cfg.hidden),
    GRAPH_CONV("gcn", cfg.hidden1, cfg.hidden2)
]
# model.set_graph_conv(convs)
model.set_graph_conv_node_type(convs)

readout = GRAPH_READOUT("mean")
model.set_graph_readout(readout)

output = nn.Linear(cfg.hidden2, cfg.embed_dim)
model.set_output_layer(output)
model.load_state_dict(torch.load("triplet_model_circuitgraph_node_type_weight_decay_from_best_to_test1_byfarbest_0.964.pt"))
model.to(cfg.device)
trainer = PairwiseGraphTrainer(cfg)
trainer.build(model)
for data in dataset:
    data = data.to(cfg.device)

cfg.trainer = 'triplet'
# cfg.epochs = 100
# cfg.epochs = 400
# trainer.triplet_train_node_type(train_loader, mining='hard', margin=0.2)
trainer.triplet_train_node_type(train_loader, mining='semihard', margin=0.2)
torch.save(model.state_dict(), "triplet_model_circuitgraph_node_type_ITC99.pt")
test_loss, test_labels, zz, test_preds = trainer.inference_node_type(test_loader)
print(len(test_pairs))
trainer.metric_calc(test_loss,  test_labels,  test_preds,  header="test ")
auc = roc_auc_score(test_labels, zz)
auc