# esmDMS Data Structures Reference

## `all_results` dict

```
all_results: dict[path_name -> result_tuple]
```

`path_name` is the basename of the embedding directory (e.g. `"my_protein"`).

---

## Result tuple layout

Both simulation and inference produce a 5-element tuple, but with different slots populated:

| Index | Simulation (`run_simulation`) | Inference (`run_inference`) |
|-------|-------------------------------|-----------------------------|
| `[0]` | `all_layer_fits` — `dict[layer -> list[float]]`, one fitness per unique sequence | `None` |
| `[1]` | `all_selection_coefficients` — `dict[layer -> np.ndarray(L,)]`, ground-truth s | `None` |
| `[2]` | `detailed_results` — see below | `detailed_results` — see below |
| `[3]` | `all_gamma_analysis` — `dict[layer -> ...]` | `None` |
| `[4]` | `all_generation_counts` — `dict[layer -> list[np.ndarray]]`, one array per generation per replicate | `None` |

`detailed_results` has identical structure in both modes.

---

## `detailed_results[layer]` — 6-element list

Produced by `_process_single_layer` (sim) and `run_inference` (real data), both packing
`mini_infer_independent_esm` output indices `[2, 3, 7, 8, 1, 5]`:

| Sub-index | Name | Shape | Description |
|-----------|------|-------|-------------|
| `[0]` | `s` | `(n_reps, L)` | Per-replicate selection coefficients |
| `[1]` | `s_joint` | `(L,)` | Joint (pooled-replicate) selection coefficients |
| `[2]` | `error_bars` | `(n_reps, L)` | Per-rep posterior std (NaN if `calc_error_bars=False`) |
| `[3]` | `s_joint_error_bars` | `(L,)` | Joint posterior std (NaN if `calc_error_bars=False`) |
| `[4]` | `icov` | list of `(L, L)` | Per-rep inverse covariance matrices |
| `[5]` | `gamma_opt` | `float` | Optimal regularization strength |

`L` = embedding dimension (e.g. 480 for ESM-2 6M, 1280 for 650M).

---

## `mini_infer_independent_esm` full return list

```
[dx, icov, s, s_joint, sel_data, gamma_opt, x_array, error_bars, s_joint_error_bars]
 [0]  [1]  [2]   [3]     [4]       [5]        [6]       [7]            [8]
```

`dx`: list of `(L,)` — allele-frequency-weighted change in embeddings, one per replicate.
`x_array`: list of `(n_seqs, L)` — embedding matrix used for inference.

---

## Embedding file formats

### Raw pickle (source of truth, may be large)
```
{embedding_path}/{basename}_embeddings.pkl
```
DataFrame columns: `ProteinSequence, Replicate, Generation, Frequency, Embedding` plus optional mutation-site metadata.
`Embedding` column: each cell is a list/array of length `n_layers`, where `emb[layer]` is a `(L,)` array.

Supported sequence embedding extraction methods:
- `mean_pool`: mean-pool real residue token embeddings, excluding CLS/EOS/padding.
- `cls`: use the ESM CLS token embedding.
- `mutation_site`: use residue-level embeddings at mutated amino-acid sites.

For `mutation_site` with `pool_mutations=false`, variants with multiple amino-acid mutations are expanded into separate dataframe rows, one per mutated site. These rows share `ProteinSequence`, `Replicate`, `Generation`, and `Frequency`, and are distinguished by `MutationSite` and `MutationSiteIndex`. With `pool_mutations=true`, mutated residue embeddings are averaged into one vector per variant.

### Compact inference format (written by `emb_df_to_inference_dfs`)
```
{embedding_path}/seq_id_map.pkl            — list[dict] or legacy list[str], length n_embedding_units
{embedding_path}/inference_metadata.pkl    — DataFrame(seq_id, Replicate, Generation, Frequency, optional MutationSite metadata)
{embedding_path}/layer{i}_seq_to_emb.pkl  — np.ndarray shape (n_embedding_units, L)
```
`load_inference_df(layer, path)` reconstructs a df with an `Embedding` column from these.

### Simulation format (written by `emb_df_to_sim_dfs`)
```
{embedding_path}/layer{i}_sim_df.pkl
```
DataFrame columns: `Embedding (L,), Rep1_PreNums, Rep2_PreNums, ...`
`load_final_df(layer, path)` reads these.

---

## Inference cache
```
{embedding_path}/inference_results[_{normalize}][_reps{ids}].pkl
```
Pickled 5-tuple matching the inference result layout above.

---

## Key access patterns used in `analysis_helpers.py`

```python
results       = all_results[path_name]          # 5-tuple
layer_fits    = results[0][layer]               # list[float] — SIM ONLY
true_s        = results[1][layer]               # np.ndarray(L,) — SIM ONLY
s             = results[2][layer][0]            # np.ndarray(n_reps, L)
s_joint       = results[2][layer][1]            # np.ndarray(L,)
gen_counts    = results[4][layer]               # list[np.ndarray] — SIM ONLY
```

`results[4]` is `None` for inference tuples — accessing it crashes.
`results[0]` and `results[1]` are `None` for inference tuples.
