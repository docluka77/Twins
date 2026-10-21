# Pipeline notebooks

The notebooks are kept essentially as they were run for the paper — including the scratch
cells and the Chinese comments — so that the reported numbers can be traced back to the
exact code that produced them. Only the hard-coded paths were replaced (repo-relative
imports, `config/`, `dataset/`).

Run them in order, with `notebooks/` as the working directory:

| Notebook | Input | Output |
|---|---|---|
| `01_dfg_generation.ipynb` | `dataset/Trust-Hub_Hardware Obfuscation Modified/**/*.v` | `DFG-new.pt` |
| `02_netlist_graph_generation.ipynb` | same | `NetlistGraph-node_type.pt` |
| `03_twins_v1_training.ipynb` | `NetlistGraph_ITC99_test.pt` | `triplet_model_v1_itc99.pt` |
| `04_twins_v2_training.ipynb` | `NetlistGraph-node_type.pt` | `triplet_model_circuitgraph_node_type_*.pt` |
| `05_gnn4ip_baseline_evaluation.ipynb` | `all_pairs.pt`, `modelparam/hw2vec.pth` | metrics, ROC curve |
| `06_embedding_visualization.ipynb` | a trained checkpoint | `embeddings.tsv`, `labels.tsv` |
| `07_itc99_generalization.py` | `NetlistGraph-node_type-ITC99.pt` | metrics |

Notes:

* The `.pt` graph datasets are large (100 MB – 250 MB); they are not committed. Regenerate
  them with stages 1–2, or point the loaders at your own copies.
* Stage 5 reproduces the GNN4IP baseline and therefore needs the GNN4IP checkpoint
  (`modelparam/hw2vec.pth`) and its pair file; the pretrained model belongs to that work.
* `models/twins_v2_netlist_graph.pt` is the checkpoint that produced the Twins-v2 row in
  the README table (F1 0.9644, accuracy 0.9929, AUC 0.9980).
