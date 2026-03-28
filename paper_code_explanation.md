# How the Papers Map to models.py

This document explains every part of `models.py` in terms of the two source papers:
- **UTSP**: *Unsupervised Learning for Solving the Travelling Salesman Problem*
- **GSN**: *Can Hybrid Geometric Scattering Networks Help Solve the Maximum Clique Problem?*

The UTSP paper describes the overall TSP-solving system. For the GNN architecture, it directly borrows the **Scattering Attention GNN (SAG)** introduced in the GSN paper (originally for a different problem, Maximum Clique). Understanding the code fully requires reading both papers together.

---

## The Big Picture

The UTSP paper frames TSP as follows:

1. Given `n` cities with 2D coordinates, build a weighted graph where edge weights encode distances.
2. Feed the graph into a GNN that outputs a soft assignment matrix `T` (shape n×n). Each entry `T[i,j]` represents the probability that city `j` is visited at step `i` in the tour.
3. Derive a heatmap `H` from `T` that assigns a probability to each edge being part of the optimal tour.
4. Minimize an unsupervised loss that encourages `H` to correspond to a short Hamiltonian Cycle.
5. Use `H` to guide a local search to find the actual tour.

The GNN used in step 2 is the SAG model from the GSN paper, adapted for TSP. The `GNN` class in `models.py` is this model. The `SCTConv` class is one layer of it.

---

## `GCN_diffusion` — Low-Pass Graph Filters

**Paper**: GSN, Section 2.2 (Graph Convolutional Networks) and Section 3.2 (Diffusion Module).

### What it does

This function computes how information flows through the graph using the GCN-style low-pass filter. The key matrix is:

```
A = (D + I)^{-1/2} * (W + I) * (D + I)^{-1/2}
```

This is the normalized adjacency matrix with self-loops added. The GSN paper (Sec. 2.2) derives this from the spectral convolution framework: Kipf & Welling's GCN applies a renormalization trick to a polynomial filter, resulting in exactly this matrix `A`. Importantly, `A` is a **low-pass filter**: it averages a node's features with its neighbors' features, making neighboring nodes look more similar.

The function computes `A^r * H` for `r = 1, 2, ...` (the `order` parameter), returning each power as a separate result. In the GSN paper this is written as:

```
f_{low,r}(H) = A^r * H
```

### How it appears in the code

```python
A_gcn = W + I_n                        # add self-loops: (W + I)
D = torch.pow(degrees, -0.5)           # D^{-1/2}  (stored as column vector for broadcasting)
A_gcn_feature = D * A_gcn_feature      # left-multiply by D^{-1/2}
A_gcn_feature = torch.matmul(A_gcn, A_gcn_feature)  # multiply by (W + I)
A_gcn_feature = torch.mul(A_gcn_feature, D)          # right-multiply by D^{-1/2}
```

Each loop iteration applies the full operation `D^{-1/2} * (W+I) * D^{-1/2} * H`, i.e. one multiplication by `A`. After `r` iterations you have `A^r * H`. With `order=2`, the function returns `[A*H, A^2*H]`.

### Why it exists

Multiplying by `A` once gathers information from direct neighbors. Multiplying by `A^2` gathers information from two-hop neighbors. These are **smooth, low-frequency** representations: nearby nodes end up with similar values. This is useful but not sufficient on its own — which is why the scattering filters below are also used.

---

## `SCT1stv2` — Band-Pass Scattering Filters

**Paper**: GSN, Section 2.3 (Graph Scattering) and Section 3.2 (Diffusion Module).

### What it does

This function computes **graph wavelet (band-pass) filters** using the geometric scattering transform. The starting point is the **lazy random walk matrix**:

```
P = 1/2 * (I + W * D^{-1})
```

Multiplying a feature vector by `P` once is like taking one step of a random walk on the graph: each node averages its own features with those of its neighbors equally. Raising `P` to a power `2^k` represents diffusion over a neighborhood of size `2^k`.

The **wavelet filters** (GSN paper, Sec. 2.3, Eq. 3) subtract two consecutive diffusion scales:

```
Ψ_0 = I - P
Ψ_k = P^{2^{k-1}} - P^{2^k},   for k ≥ 1
```

Each `Ψ_k` detects **changes** between neighborhoods at scale `2^{k-1}` and scale `2^k`. This is analogous to a bandpass filter in signal processing (or an edge detector in image processing): it picks up information that is present at one scale but not at the next, capturing local structure.

### How it appears in the code

```python
P = 0.5 * I + 0.5 * W * D^{-1}           # lazy random walk
```

The loop applies `P` repeatedly:

```python
feature_p = 0.5 * feature_p + 0.5 * W_D_inv_x   # one application of P
```

The `scale_list` with `order=4` is `[0, 1, 3, 7, 15]`, which correspond to after 1, 2, 4, 8, 16 applications of `P`, i.e. `P^1*x, P^2*x, P^4*x, P^8*x, P^16*x`. These are captured into `sct_diffusion_list`.

The wavelet features are then the differences between consecutive scales:

```python
sct_feature1 = P^1*x  - P^2*x    # Ψ_1 * x  (scale 1, sensitive to 1-2 hop differences)
sct_feature2 = P^2*x  - P^4*x    # Ψ_2 * x  (scale 2, sensitive to 2-4 hop differences)
sct_feature3 = P^4*x  - P^8*x    # Ψ_3 * x  (scale 3)
sct_feature4 = P^8*x  - P^16*x   # Ψ_4 * x  (scale 4, sensitive to long-range structure)
```

### Why it exists

The scattering filters are **band-pass**: they respond strongly where the graph signal *changes* between two scales. This gives the network the ability to detect boundaries — for example, the border between nodes that belong to a solution and nodes that do not. The GSN paper motivates this explicitly: a plain GCN (low-pass only) cannot distinguish a node bordering a clique from an actual clique member because it blurs everything together (**oversmoothing**). The UTSP paper repeats this motivation for TSP: a non-smooth heatmap is critical, and scattering filters help produce it.

The `** moment` applied in `SCTConv.forward` is the pointwise nonlinearity from the scattering cascade (GSN paper, Eq. 4). Taking `|x|^moment` (with `moment=1` this is just `|x|`) prevents cancellation from positive and negative wavelet coefficients, analogous to a rectifier in a wavelet scattering network.

---

## `SCTConv` — One Scattering Attention Layer

**Paper**: GSN, Section 3.2 (Diffusion Module).

### What it does

`SCTConv` is one full layer of the SAG model. It takes node features `X` and the adjacency matrix `adj`, runs them through all 6 filters (2 low-pass + 4 band-pass), and combines the results using a learned attention mechanism.

### The filter set

The 6 filters in each layer are:

| Variable  | Filter                  | Type       | Paper notation     |
|-----------|-------------------------|------------|--------------------|
| `h_A`     | `A^1 * H`               | Low-pass   | `f_{low,1}(H)`     |
| `h_A2`    | `A^2 * H`               | Low-pass   | `f_{low,2}(H)`     |
| `h_sct1`  | `|Ψ_1 * H|`             | Band-pass  | `f_{band,1}(H)`    |
| `h_sct2`  | `|Ψ_2 * H|`             | Band-pass  | `f_{band,2}(H)`    |
| `h_sct3`  | `|Ψ_3 * H|`             | Band-pass  | `f_{band,3}(H)`    |
| `h_sct4`  | `|Ψ_4 * H|`             | Band-pass  | `f_{band,4}(H)`    |

Using both types is what makes this a *hybrid* model. The low-pass filters capture smooth neighborhood context; the band-pass filters capture local contrast and structure at multiple scales.

### The attention mechanism

The GSN paper (Sec. 3.2, Eq. 5-7) describes how to combine the 6 filter outputs using attention. For each filter `f`, a score is computed for each node:

```
s_f(v) = ReLU([H || H_f]) * a
```

where `[H || H_f]` is the concatenation of the original features with the filtered features, and `a` is a learned attention vector (shared across all nodes). This measures how useful filter `f`'s output is for each node individually — different nodes may find different filters more informative.

The scores are normalized across all 6 filters using softmax:

```
alpha_f(v) = softmax over filters of s_f(v)
```

The final output is a weighted sum:

```
H_agg = sum over f of alpha_f(v) * H_f(v)
```

### How it appears in the code

```python
# Compute attention scores for each filter (shape: [B, 6, N, 1])
a_input = cat([h || h_f for each filter f]).view(B, 6, N, -1)
e = matmul(relu(a_input), self.a).squeeze(3)   # [B, 6, N]

# Normalize across the 6 filters per node
attention = softmax(e, dim=1).view(B, 6, N, -1)  # [B, 6, N, 1]

# Weighted combination
h_all = cat([h_A, h_A2, h_sct1, ..., h_sct4], dim=1)  # [B, 6, N, F]
h_prime = mean(attention * h_all, dim=1)               # [B, N, F]
```

The `mean` at the end gives the aggregated representation. It is then passed through two linear layers with LeakyReLU to produce the layer output.

Note: the GSN paper applies the nonlinearity `sigma` **before** multiplying with the attention vector `a` (inside `relu(a_input)`), rather than after. The paper (Sec. 3.2) explicitly notes this departs from the original GAT attention and is shown to be more expressive.

---

## `GNN` — The Full Model

**Paper**: Both papers. The architecture is from GSN (Sections 3.1–3.3). The usage (input/output format, what the output means) is from UTSP (Section 3.1–3.2).

### Embedding module (GSN Sec. 3.1)

```python
self.in_proj = nn.Linear(input_dim, hidden_dim)
```

The raw node features (city coordinates `(x, y)` in UTSP; graph statistics in GSN) are first projected to a `hidden_dim`-dimensional space. In TSP, `input_dim=2`.

### Diffusion module — the stacked SCTConv layers (GSN Sec. 3.2)

```python
self.convs = nn.ModuleList([SCTConv(hidden_dim) for _ in range(n_layers)])
```

Each `SCTConv` layer is one iteration of the diffusion module. After each layer, the output is concatenated to a growing `hidden_states` tensor:

```python
hidden_states = cat([hidden_states, X], dim=-1)
```

This accumulates `H^0, H^1, ..., H^K` — the **readout list** `R` in the GSN paper (Sec. 3.2). The idea is that each layer captures information at a different depth, and keeping all of them lets the output module draw on representations at multiple levels of abstraction.

### Output module (GSN Sec. 3.3)

```python
self.mlp1 = nn.Linear(hidden_dim * (1 + n_layers), hidden_dim)
self.mlp2 = nn.Linear(hidden_dim, output_dim)
self.m = nn.Softmax(dim=1)
```

The concatenated readouts (shape `[B, n, hidden_dim*(1+n_layers)]`) are compressed back to `output_dim` via two MLP layers. Then a **column-wise softmax** is applied.

In the UTSP paper (Sec. 3.1), this softmax produces the soft indicator matrix `T`:

```
T[i, j] = exp(S[i, j]) / sum_k exp(S[k, j])
```

Each column of `T` sums to 1, making `T` a stochastic matrix. The entry `T[i, j]` is interpreted as the probability that city `i` is visited at position `j` in the tour.

In the GSN paper (Sec. 3.3) the equivalent step uses min-max normalization to produce a probability vector `p` over nodes. The UTSP adapts this for TSP by outputting an n×n matrix (one probability distribution per tour position) and using softmax instead of min-max normalization.

---

## The Loss Function and Heatmap (UTSP only)

**Paper**: UTSP, Sections 3.3–3.5.

These are not part of `models.py` but are essential context for understanding what `GNN` is trained to do.

### Heatmap construction (UTSP Sec. 3.3)

From the soft indicator matrix `T`, a heatmap `H` is built:

```
H = sum_{t=1}^{n-1} p_t * p_{t+1}^T + p_n * p_1^T
```

where `p_t` is the `t`-th column of `T`. Intuitively: if city `i` is likely at position `t` and city `j` is likely at position `t+1`, then edge `(i,j)` gets high probability. The sum over all consecutive position pairs gives the full heatmap. The paper proves that if `T` is a permutation matrix (one `1` per row and column), `H` encodes exactly one Hamiltonian Cycle.

### Unsupervised loss (UTSP Sec. 3.4)

```
L = λ₁ * ||row_sums(T) - 1||² + λ₂ * trace(H) + sum_{i,j} D[i,j] * H[i,j]
```

- **Term 1 (row-wise constraint)**: Softmax already ensures each column of `T` sums to 1. This term pushes each *row* to also sum to 1, making `T` doubly stochastic — as close to a permutation matrix as possible.
- **Term 2 (no self-loops)**: The diagonal of `H` encodes self-loops (city `i` immediately followed by itself). This term penalizes that.
- **Term 3 (shortest path)**: This is the expected total tour length under the heatmap `H`. Minimizing it encourages the predicted tour to be short.

The first two terms enforce the Hamiltonian Cycle structure; the third enforces minimality. No labelled tour data is needed — the loss is purely unsupervised.

---

## Summary: Which Code Comes From Which Paper

| Code | Source | Paper Section |
|------|--------|---------------|
| `GCN_diffusion` | GSN | Sec. 2.2 (GCN), Sec. 3.2 low-pass filter `f_{low,r}` |
| `SCT1stv2` | GSN | Sec. 2.3 (Graph Scattering), wavelet matrix `Ψ_k` |
| `SCTConv` | GSN | Sec. 3.2 (Diffusion Module), attention mechanism |
| `GNN.in_proj` | GSN | Sec. 3.1 (Embedding Module) |
| `GNN.convs` + readout concatenation | GSN | Sec. 3.2 (Diffusion Module), readout list `R` |
| `GNN.mlp1/mlp2` | GSN | Sec. 3.3 (Output Module) |
| `GNN.m` (Softmax) | UTSP | Sec. 3.1, column-wise softmax to produce `T` |
| Heatmap `H` construction | UTSP | Sec. 3.3, `T → H` transformation |
| Unsupervised loss `L` | UTSP | Sec. 3.4, three-term loss |
| Motivation for scattering (non-smoothness) | Both | GSN Sec. 1, UTSP Sec. 4 |
