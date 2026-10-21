# Twins

**Twins: Hardware Similarity Evaluation Framework Using Graph Neural Network**

Twins decides whether two hardware designs are *functionally the same* by encoding each
design into a graph embedding and measuring the distance between embeddings. It targets
tasks such as IP piracy detection and gate-level netlist reverse engineering, where a
candidate module has to be matched against a known-good reference.

The framework ships two variants:

| Variant | Encoder | Graph input |
|---|---|---|
| **Twins-v1** | Two GCN layers + mean-pooling readout + FC projection | Data flow graph (DFG) or netlist graph |
| **Twins-v2** | Node-independent GCN layers — one weight set per node type, selected by node type | Netlist graph |

Both are trained with `TripletMarginLoss` (cosine similarity, margin `0.2`) and **semi-hard
negative mining performed online**, i.e. every graph in a batch is encoded exactly once
and triplets are built from the embeddings. This reduces the training complexity from
`O(n²)` (the pair/triplet re-encoding used by GNN4IP and other similarity-learning work)
to `O(n)`.

## Repository layout

```
Twins/
├── hw2vec/                     # core library: graph construction + GNN encoders + trainers
│   ├── hw2graph.py             #   verilog → PyVerilog DFG / CircuitGraph netlist graph
│   ├── utilities.py            #   DataProcessor: NetworkX → PyG conversion, node features
│   ├── config.py               #   YAML-driven configuration object
│   ├── configs/                #   default configs shipped with the upstream package
│   └── graph2vec/
│       ├── models.py           #   GRAPH2VEC: GCN stacks, node-independent stacks, readout
│       └── trainers.py         #   PairwiseGraphTrainer: triplet_train / triplet_train_node_type,
│                               #   inference, metric calculation
├── config/
│   └── example_gnn4ip.yaml     # hyper-parameter file used by the pipeline scripts
├── notebooks/                  # the pipeline, in execution order
│   ├── 01_dfg_generation.ipynb
│   ├── 02_netlist_graph_generation.ipynb
│   ├── 03_twins_v1_training.ipynb
│   ├── 04_twins_v2_training.ipynb
│   ├── 05_gnn4ip_baseline_evaluation.ipynb
│   ├── 06_embedding_visualization.ipynb
│   └── 07_itc99_generalization.py
├── models/
│   └── twins_v2_netlist_graph.pt   # trained Twins-v2 checkpoint (netlist graph, 10-dim features)
├── requirements.txt
└── LICENSE
```

## Pipeline

Each stage reads the artifacts of the previous one; run the notebooks **from inside
`notebooks/`** (they resolve `../hw2vec`, `../config` and `../dataset` relative to the
working directory).

| # | Stage | Produces |
|---|---|---|
| 1 | `01_dfg_generation.ipynb` — parse Verilog with PyVerilog, merge the duplicated node spellings that PyVerilog emits (`c17.N16` vs `c17_N16`), one-hot the 17 node types | `DFG-new.pt` |
| 2 | `02_netlist_graph_generation.ipynb` — parse the gate-level netlist with CircuitGraph, collapse intermediate signals, one-hot the 10 node types (`input`, `and`, `nand`, `or`, `nor`, `not`, `buf`, `xor`, `xnor`, `output`) | `NetlistGraph-node_type.pt` |
| 3 | `03_twins_v1_training.ipynb` — train and evaluate the basic model | `triplet_model_v1_*.pt` |
| 4 | `04_twins_v2_training.ipynb` — train and evaluate the node-independent model | `triplet_model_*_node_type_*.pt` |
| 5 | `05_gnn4ip_baseline_evaluation.ipynb` — the GNN4IP baseline (DFG, 37-dim features, pairwise training) re-evaluated under our graph-level train/test split | metrics, ROC/AUC |
| 6 | `06_embedding_visualization.ipynb` — UMAP projection of the learned embeddings | `embeddings.tsv`, `labels.tsv` |
| 7 | `07_itc99_generalization.py` — the same model trained/evaluated on the ITC'99 obfuscation set | metrics |

Training-only hyper-parameters for Twins-v1 (from the paper): 200 epochs, batch size 64,
node dimensions 64 → 128, graph embedding 256, Adam (`lr=0.01`, `weight_decay=5e-4`),
ReLU, dropout 0.1, triplet margin 0.2, cosine distance, semi-hard mining.
Twins-v2 uses 300 epochs, `lr=0.005` halved every 100 epochs, `weight_decay=5e-3`.

## Dataset

305 combinational circuits, all as technology-independent gate-level Verilog:
10 circuits from **ISCAS'85** and 295 obfuscated variants from the **Trust-Hub Obfuscation
Benchmark** (RN, SL, CS, CY, NR, NS, NC, BR, BS, BE techniques).

The notebooks expect the raw circuits at

```
dataset/Trust-Hub_Hardware Obfuscation Modified/<circuit>/<variant>/topModule.v
```

e.g. `dataset/Trust-Hub_Hardware Obfuscation Modified/C432/c432-BE280/topModule.v`.
The circuit name is the second-to-last path component and the variant the third-to-last,
which is how labels are derived. Set `DATASET_DIR` in the first cells of stages 1 and 2 if
your copy lives elsewhere.

## Environment

Validated with Python 3.6–3.10, PyTorch + PyTorch Geometric on a single NVIDIA V100S.
`pip install -r requirements.txt` pulls the modern equivalents; the original experiments
ran on older pinned versions, so exact metric reproduction may require matching that
stack.

## Results (report in brief)

| Method | Graph | Accuracy | Precision | F1 | AUC | Training time | Inference time |
|---|---|---|---|---|---|---|---|
| GNN4IP | DFG | — | — | — | — | 38 h 1 m 21 s | 140.0 s |
| Twins-v1 | DFG | 0.9678 | — | 0.8577 | — | 37.1 s | 1.7 s |
| Twins-v1 | Netlist graph | 0.9874 | 0.9312 | 0.9387 | — | 20.1 s | 1.6 s |
| Twins-v2 | Netlist graph | 0.9929 | 0.9832 | 0.9644 | 0.9980 | 8 m 49.2 s | 3.7 s |

Netlist graphs are roughly half the size of the equivalent DFG (4,944 vs 8,998 nodes on
average) and take 78% less time to extract, which is what makes them the better input for
this task. Full tables, ablations and discussion are in the paper.

## Citation

```bibtex
@article{he2026twins,
  title   = {Twins: Hardware Similarity Evaluation Framework Using Graph Neural Network},
  author  = {He, Yiran and Shen, Haihua and Jiang, Zirui and Li, Shan and Ji, Xiao and Li, Huawei},
  journal = {ACM Transactions on Design Automation of Electronic Systems (TODAES)},
  year    = {2026}
}
```

## Acknowledgement and licence

The graph-construction layer (`hw2vec/hw2graph.py`, `hw2vec/utilities.py`,
`hw2vec/graph2vec/`) is derived from [HW2VEC](https://github.com/louisccc/hw2vec) by
Shih-Yuan Yu et al., released under the MIT licence; the original copyright notice is
kept in `LICENSE`. The netlist-graph extraction, the node-independent encoder, the online
mining trainers and the evaluation pipeline are part of this work and are released under
the same MIT terms.
