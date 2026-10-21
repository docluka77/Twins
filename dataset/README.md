# Dataset

305 combinational circuits, in technology-independent gate-level Verilog:

| Source | Circuits | Notes |
|---|---|---|
| ISCAS'85 | 10 | C432, C499, C880, C1355, C1908, C2670, C3540, C5315, C6288, C7552 |
| Trust-Hub Obfuscation Benchmark | 295 | obfuscated variants of the ten ISCAS'85 circuits |

Obfuscation techniques covered: RN, SL, CS, CY, NR, NS, NC, BR, BS, BE.
The original circuits come from [Trust-Hub](https://trust-hub.org) and the ISCAS'85
benchmark distribution; both are public and are not redistributed here.

## Expected layout

```
dataset/Trust-Hub_Hardware Obfuscation Modified/<circuit>/<variant>/topModule.v
```

For example:

```
dataset/Trust-Hub_Hardware Obfuscation Modified/C432/c432-BE280/topModule.v
dataset/Trust-Hub_Hardware Obfuscation Modified/C7552/c7552-RN1050/topModule.v
```

Stage 2 does not read the file contents for labelling — it takes the third-to-last path
component as the circuit name and the second-to-last as the variant name:

* the **circuit name** (`C432`, `c7552`, …) becomes the class label used by the triplet
  loss, so obfuscated variants of the same circuit share a label;
* the **variant name** (`c432-BE280`, …) is stored as `hw_name` on the graph and is used
  when exporting embeddings.

Keep `topModule.v` as the file name, and keep the directory naming convention, otherwise
both the labelling and the graph extraction will silently mis-associate designs.

## Processed artifacts

| File | Produced by | Size |
|---|---|---|
| `DFG-new.pt` | `01_dfg_generation.ipynb` — 17-dimensional one-hot node features | ~243 MB |
| `NetlistGraph-node_type.pt` | `02_netlist_graph_generation.ipynb` — 10-dimensional one-hot node features | ~108 MB |

Both are plain `torch.save` lists of `torch_geometric.data.Data`. Only the netlist-graph
file is required for Twins-v2.
