# Comparison: `neural/` Implementation vs. UTSP Reference

## Similarities (What Is the Same)

| Aspect | Both Implementations |
|--------|---------------------|
| **Model architecture** | ScatteringAttentionGNN with identical structure: Embedding → stacked ScatteringConvolution layers → concatenated readout → MLP + Softmax |
| **ScatteringConvolution** | Same 6 filters: 2 GCN low-pass (A\*H, A²\*H) + 4 band-pass scattering wavelets at dyadic scales [0,1,3,7,15]. Same attention mechanism over filters |
| **Scattering math** | Identical lazy random walk P=0.5(I+WD⁻¹), same scale\_list `[2^i-1]`, same wavelet differences |
| **GCN diffusion** | Identical D^{-0.5}(W+I)D^{-0.5} symmetric normalization |
| **Loss function** | Same 3-term unsupervised loss: row-sum constraint + diagonal/self-loop penalty + distance-weighted heatmap cost |
| **Heatmap computation** | Identical: `H = T @ roll(T^T, -1, 1)` — outer product sum of shifted columns |
| **Adjacency from distances** | Both: `adj = exp(-distance / temperature)` with diagonal zeroed out |
| **Optimizer** | Adam + StepLR + gradient clipping to norm 1.0 |
| **Inference pipeline** | Both: GNN → heatmap → top-k extraction → MCTS C++ binary |
| **Softmax** | Both apply `Softmax(dim=1)` on the output module (row-wise normalization) |

---

## Differences

| Aspect | UTSP Reference | `neural/` Implementation | Notes |
|--------|---------------|--------------------------|-------|
| **Input features** | Raw 2D coordinates `[B,N,2]`, `input_dim=2` | Two modes: `"coords"` (2D coordinates) or `"node_stats"` (8 distance statistics per node). Both available, selected via config. | `coords` mode matches UTSP. `node_stats` computes features from raw distances (mean, std, min, quartiles, max, k-nearest mean). |
| **Distance type** | Euclidean: computed from coordinates via `scipy.spatial.distance_matrix` | Pre-computed real-world travel times (non-Euclidean, symmetric) loaded from HDF5 | Triangle inequality may not hold. Distances have different distribution than Euclidean. |
| **Distance normalization** | Coordinates rescaled by `args.rescale`, distances derived from rescaled coords | `adj / mean(off_diagonal)` — normalizes distances so typical value ≈ 1 | Different approach. UTSP rescales *coordinates* (indirectly scaling distances). `neural/` normalizes the *distance matrix* directly. |
| **Coordinate rescaling** | `coords = rescale * (coords - mean)`, with `rescale` as hyperparameter (1.0-4.0) | ~~`RESCALE_COORDS` defined but never applied~~ **Fixed:** removed entirely. Not applicable in non-Euclidean setting — geographic lat/lon have no meaningful relationship to travel time scale. | Resolved. |
| **Defaults** | hidden=64, layers=3, lr=3e-3, λ₁=10 | ~~hidden=256, layers=4, lr=5e-2, λ₁=20~~ **Fixed:** defaults now hidden=64, layers=3, lr=3e-3, λ₁=10 (matching UTSP). Actual values per problem size are determined by hyperparameter sweep and stored in `neural/config/best/<size>.yaml`. | Resolved. |
| **Epochs** | 20 (train.py default, but saves checkpoints from epoch 200+ — implying actual runs are much longer) | 300 | Not a real difference. Both likely train for ~300 epochs. |
| **BatchNorm** | `self.bn0 = nn.BatchNorm1d(input_dim)` defined but **never called** in forward | No BatchNorm (commented out in EmbeddingModule) | Same net effect — neither actually uses BatchNorm. |
| **GCN diffusion order** | Computes 3 orders (A, A², A³) but only uses A, A² in attention (skips A³) | Computes 2 orders only (A, A²) | Correct simplification. UTSP wastes computation on A³. |
| **Data format** | NumPy `.npy` files: `[N_instances, N_nodes, 2]` coords + solutions | HDF5 `.h5` container: upper-tri distance matrix + coords per problem | `neural/` is more space-efficient and supports concurrent reads. |
| **Data split** | Fixed: 2000 train, rest val | 70/20/10 random split | More standard approach in `neural/`. |
| **Loss normalization** | `loss / len(batch)` manually | Each term averaged by batch\_size inside `unsupervised_loss` | Equivalent in intent, cleaner in `neural/`. |
| **Loss: distance term** | Uses `distance_matrix` (raw Euclidean distances) | Uses normalized distance matrix (typical value ≈ 1) | After normalization, both have similar absolute scale (values ≈ 1). However, the *distribution* differs: UTSP's Euclidean distances from uniform random points are smoothly spread, while road network travel times are skewed (many short trips, fewer long ones). This changes the loss landscape and gradient behavior, requiring λ₁/λ₂ re-tuning per problem size via sweep. |
| **Inference: distance to MCTS** | Euclidean distances (float) derived from coordinates | Raw integer travel times cast to `int64` | Fine for travel times in seconds. |
| **Inference: adj symmetrization** | Already symmetric (Euclidean) | `np.maximum(adj_raw, adj_raw.T)` — forces symmetry | Necessary for real-world data with one-directional storage. |
| **Top-k (REC\_NUM)** | 10-20 (varies by problem size) | 20 (hardcoded) | Must match the compiled C++ binary's `Rec_Num`. |
| **Config management** | Hardcoded per-size scripts | Per-size YAML configs in `neural/config/best/<size>.yaml`, loaded by `neural/config/loader.py` for both training and inference. | Enables systematic hyperparameter management across problem sizes. |

---

## Remaining Considerations

1. **Temperature-distance scale interaction**: Temperature values found by sweep (3.9–5.0) are higher than UTSP's defaults (2.0–3.5). This makes sense — with normalized distances (mean ≈ 1), a higher temperature is needed to produce a similar adjacency distribution as UTSP gets with raw Euclidean distances.

2. **Loss balance across sizes**: The row constraint term grows with problem size (1.8 at N=25, 4.7 at N=50, 10.9 at N=100) while λ₁ stays low (3–6 from sweeps). The distance term dominates at ~90% of total loss, which is the desired behavior — the model primarily optimizes tour cost.

3. **Coords vs node\_stats**: Hyperparameter sweeps consistently selected `coords` over `node_stats` for sizes 25, 50, and 100. This suggests geographic coordinates carry useful signal even for non-Euclidean road network distances. This may change at larger sizes where road topology diverges more from straight-line geometry.
