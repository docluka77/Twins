# Pipeline notebooks

Run in order, with `notebooks/` as the working directory.

| Notebook | Input | Output |
|---|---|---|
| `01_dfg_generation.ipynb` | `dataset/Trust-Hub_Hardware Obfuscation Modified/**/*.v` | `DFG-new.pt` |
| `02_netlist_graph_generation.ipynb` | same | `NetlistGraph-node_type.pt` |
| `03_twins_v1_training.ipynb` | netlist graph dataset | `triplet_model_v1_*.pt` |
| `04_twins_v2_training.ipynb` | `NetlistGraph-node_type.pt` | `triplet_model_*_node_type_*.pt` |
| `05_gnn4ip_baseline_evaluation.ipynb` | `all_pairs.pt`, GNN4IP checkpoint | metrics, ROC curve |
| `06_embedding_visualization.ipynb` | a trained checkpoint | `embeddings.tsv`, `labels.tsv` |
| `07_itc99_generalization.py` | `NetlistGraph-node_type-ITC99.pt` | metrics |

The notebooks are kept close to how they were run, including scratch cells and Chinese
comments. The `.pt` graph datasets are large (100–250 MB) and are not committed —
regenerate them with stages 1–2.
