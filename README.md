# Twins

**Twins: Hardware Similarity Evaluation Framework Using Graph Neural Network**

Twins decides whether two hardware designs are functionally the same: a GNN encodes each
design into a graph embedding, and similarity is read off the distance between embeddings.
The netlist is modelled as a **netlist graph** — gates and I/O as nodes, wires as edges —
rather than a data flow graph, which keeps the graph roughly half the size. Training uses
`TripletMarginLoss` with **online semi-hard negative mining**, so every graph is encoded
once per iteration instead of once per pair.

Two encoders are included: `Twins-v1`, a shared GCN stack, and `Twins-v2`, which keeps a
separate weight set for each node type.

## Contents

```
hw2vec/     core library: graph construction, GNN encoders, triplet trainers
config/     hyper-parameter file used by the pipeline
notebooks/  the pipeline, numbered in execution order
```

## Usage

```bash
pip install -r requirements.txt
cd notebooks          # the notebooks resolve ../hw2vec, ../config and ../dataset
```

Run `01` → `02`, then `03` (Twins-v1) or `04` (Twins-v2). `05`–`07` cover the GNN4IP
baseline comparison, the embedding visualisation and the ITC'99 run.

Raw circuits are expected at
`dataset/Trust-Hub_Hardware Obfuscation Modified/<circuit>/<variant>/topModule.v`;
see `dataset/README.md` for the layout.

## Dataset

305 gate-level combinational circuits: 10 from ISCAS'85 and 295 obfuscated variants from
the Trust-Hub Obfuscation Benchmark. Both are public and are not redistributed here.

## Citation

```bibtex
@article{he2026twins,
  title   = {Twins: Hardware Similarity Evaluation Framework Using Graph Neural Network},
  author  = {He, Yiran and Shen, Haihua and Jiang, Zirui and Li, Shan and Ji, Xiao and Li, Huawei},
  journal = {ACM Transactions on Design Automation of Electronic Systems (TODAES)},
  year    = {2026}
}
```

## Licence

MIT. The graph-construction layer derives from
[HW2VEC](https://github.com/louisccc/hw2vec); its copyright notice is kept in `LICENSE`.
