# popDMS ESM-DMS Analysis Pipeline

This document describes the pipeline from raw DMS count data to ESM2 embeddings, transformed feature spaces, popDMS inference, and analysis outputs.

## 1. Sequence Reconstruction

Input data lives under:

```text
/Users/dylanwells/popDMS/esmDMS/data/raw_data
```

Embedding jobs are configured with files like:

```text
embedding_scripts/embed_config_TpoR.json
embedding_scripts/embed_config_Ube4b.json
embedding_scripts/embed_config_BRCA1.json
embedding_scripts/embed_config_BF520.json
embedding_scripts/embed_config_BG505.json
```

`embedding_scripts/embed_sequences.py` supports two count formats:

- Codon-count datasets with pre-selection and post-selection files.
- MaveDB nucleotide-count datasets with `hgvs_nt` variants and replicate/generation count columns.

The script reconstructs each mutant protein sequence and records:

- `ProteinSequence`
- `Replicate`
- `Generation`
- `Frequency`
- `MutationSites` when mutation positions can be inferred

For MaveDB substitutions, `MutationSites` contains 0-indexed amino-acid positions whose amino acid differs from the translated reference.

## 2. ESM2 Embedding Extraction

The embedding script computes hidden states from the configured ESM2 model and supports three extraction methods.

### Mean-Pool

```bash
python embedding_scripts/embed_sequences.py embedding_scripts/embed_config_TpoR.json TpoR_embeddings.pkl /tmp/esm_embed --embedding_method mean_pool
```

This is the historical baseline. It averages real residue token embeddings and excludes CLS/EOS/padding tokens.

### CLS Token

```bash
python embedding_scripts/embed_sequences.py embedding_scripts/embed_config_TpoR.json TpoR_cls_embeddings.pkl /tmp/esm_embed --embedding_method cls
```

This uses the ESM2 CLS token hidden state as a sequence-level representation.

### Mutation-Site Residue Embeddings

```bash
python embedding_scripts/embed_sequences.py embedding_scripts/embed_config_TpoR.json TpoR_mutation_site_embeddings.pkl /tmp/esm_embed --embedding_method mutation_site
```

This uses the residue-level hidden state at each mutated amino-acid site. For variants with two or more amino-acid mutations, the default is one dataframe row per mutated residue:

```text
ProteinSequence, MutationSites, MutationSite, MutationSiteIndex, Embedding, Frequency
```

Rows for the same parent protein share the same frequency trajectory but differ in `MutationSite` and `MutationSiteIndex`.

To pool multiple mutated residues into one vector per protein:

```bash
python embedding_scripts/embed_sequences.py embedding_scripts/embed_config_TpoR.json TpoR_mutation_site_pooled_embeddings.pkl /tmp/esm_embed --embedding_method mutation_site --pool_mutations
```

## 3. Batch Embedding Jobs

Submit all five walkthrough datasets with 10 workers per dataset:

```bash
bash job_scripts/submit_five_dataset_embeddings.sh
```

Submit CLS embeddings:

```bash
bash job_scripts/submit_five_dataset_embeddings.sh cls
```

Submit unpooled mutation-site embeddings:

```bash
bash job_scripts/submit_five_dataset_embeddings.sh mutation_site
```

Submit pooled mutation-site embeddings:

```bash
bash job_scripts/submit_five_dataset_embeddings.sh mutation_site 10 --pool_mutations
```

The script submits a 10-task Slurm array for each dataset:

- `TpoR`
- `Ube4b`
- `BRCA1`
- `BF520`
- `BG505`

It also submits a dependent merge job for each dataset using `embedding_scripts/merge_embedding_chunks.py`.

## 4. Raw Embedding Pickles

Embedding jobs write pickles under:

```text
data/sequence_data
```

Examples:

```text
TpoR_embeddings.pkl
TpoR_cls_embeddings.pkl
TpoR_mutation_site_embeddings.pkl
TpoR_mutation_site_pooled_embeddings.pkl
```

The `Embedding` column stores one array per variant. For sequence-level methods, each array has shape:

```text
(n_layers, embedding_dim)
```

For unpooled mutation-site embeddings before expansion, each array has shape:

```text
(n_mutated_sites, n_layers, embedding_dim)
```

The embedding script expands these into one row per mutated site before saving long-format output.

## 5. Compact Inference Files

`mega_analysis.emb_df_to_inference_dfs(...)` converts raw embedding pickles into compact per-layer files:

```text
seq_id_map.pkl
inference_metadata.pkl
layer0_seq_to_emb.pkl
...
layer33_seq_to_emb.pkl
```

For mean-pool and CLS embeddings, each `seq_id` usually corresponds to one protein sequence.

For unpooled mutation-site embeddings, each `seq_id` corresponds to an embedding unit:

```text
(ProteinSequence, MutationSiteIndex, MutationSite)
```

This preserves separate residue-level embeddings for multi-mutant variants.

## 6. Feature-Space Transforms

After extraction, embeddings can be transformed before popDMS inference. The transform interface is:

```python
transform.fit(embeddings)
features = transform.transform(embeddings)
```

Implemented transforms:

- `IdentityTransform`
- `PCATransform`
- `SparsePCATransform`
- `SparseAutoencoderTransform`
- `ICATransform`

The downstream popDMS inference step is unchanged. It receives transformed embeddings and infers selection coefficients in that transformed basis.

## 7. Class-Based Analysis API

The main analysis object is `EmbeddingAnalysisDataset`:

```python
from mega_analysis import EmbeddingAnalysisDataset
from embedding_transforms import PCATransform

dataset = EmbeddingAnalysisDataset(
    "TpoR",
    "/Users/dylanwells/popDMS/esmDMS/data/sequence_data/TpoR",
    inference_cfg={"layers": [33], "normalize": "by_layer_dim"},
)

result = dataset.run_inference(PCATransform(variance_threshold=0.95))
summary = result.summary()
```

For all registered transforms:

```python
results = dataset.run_inference_many()
```

For simulations:

```python
sim_result = dataset.run_simulation(PCATransform(variance_threshold=0.95))
dataset.plot_standard(sim_result, ["fitness", "sel_coeffs"], "plots/TpoR_pca")
```

## 8. Standard Analysis Options

The CLI and notebook both dispatch through `mega_analysis.ANALYSES`.

Single-dataset examples use `TpoR`:

- `fitness`
- `sel_coeffs`
- `cross_replicate_consistency`
- `shuffled_frequencies`
- `popDMS_comparison`
- `popDMS_enrichment_comparison`
- `embedding_transform_comparison`

Multi-dataset shuffled consistency examples use:

- `BG505`
- `BF520`

Some registry flags are explicit placeholders:

- `gamma`
- `sparsify`
- `eigenvalue_distribution`

## 9. Outputs

Typical outputs include:

- `inference_results*.pkl`
- `transform_inference_results*.pkl`
- `embedding_transform_summary.csv`
- cross-replicate heatmaps and scatter plots
- shuffled negative-control plots
- popDMS/enrichment comparison plots
- notebook result pickles and CSV summaries under `walkthrough_outputs`

## 10. Interpretation

The final inferred vector `s` always lives in the feature basis passed to popDMS.

- Mean-pool or CLS with identity transform: coefficients are in raw ESM2 embedding coordinates.
- PCA/SPCA/ICA: coefficients are in component space and can be projected back to ESM2 coordinates when linear.
- SAE: coefficients are in sparse latent activation space.
- Mutation-site embeddings: coefficients describe residue-local ESM2 feature variation rather than whole-sequence pooled variation.
