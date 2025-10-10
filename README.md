# CrysToGraph
Crystals with Transformers on Graph

## Direction-aware targets

Some property prediction tasks come with an associated in-plane direction
(e.g. the \(x\) or \(y\) axis chosen while constructing a surface slab).  The
dataset and model utilities now support injecting that directional metadata as
an additional input feature so that the finetuning model can condition its
predictions on the chosen orientation.

1. **Collect the per-structure direction vectors.**  Suppose you have already
   computed a pair of lattice vectors `(x_sup, y_sup)` via the helper utilities
   in your surface construction workflow.  Stack the components of the desired
   direction (for instance `[1, 0, 0]` for \(x\) or `[0, 1, 0]` for \(y\)) in the
   order of the processed dataset.
2. **Attach them to the dataset.**  After initialising a
   `ProcessedDGLCrystalDataset`, call
   `set_directional_features(vectors, key='direction', normalize=True)`.  The
   vectors will be broadcast to all nodes in the corresponding graph so they
   can be pooled together with the atom embeddings.【F:CrysToGraph/data/crystal.py†L247-L306】
   If you need to temporarily opt out (for instance when reusing the dataset
   for a pretraining stage) call `enable_directional_features(False)` before
   sampling batches.【F:CrysToGraph/data/crystal.py†L389-L408】
3. **Enable the model to consume the feature.**  When constructing the
   `Finetuning` module pass `direction_dim=<vector length>` and (optionally) a
   `direction_key` if you stored the feature under a different name.  The
   finetuning head will project the pooled direction vector and fuse it with
   the crystal representation before the fully-connected layers.【F:CrysToGraph/model/NN.py†L200-L257】

This workflow lets you combine scalar targets with categorical or vectorial
metadata describing the orientation of the slab without modifying the rest of
the training pipeline.  Direction vectors are intentionally ignored during
contrastive pretraining so that pretraining objectives remain agnostic to the
choice of orientation; the metadata is only fused in during finetuning for the
downstream target.【F:CrysToGraph/data/crystal.py†L569-L588】
