# Theory Behind `esmDMS.py`

This document explains the mathematical and modeling ideas implemented in
`esmDMS.py`, with the selection inference details from `popDMS.py` because
`esmDMS.py` delegates the actual selection-coefficient calculation to that
module. The code is built around a single pipeline:

1. Reconstruct protein variants and their observed count time series.
2. Embed each protein sequence with an ESM-family protein language model.
3. Convert high-dimensional residue hidden states into sequence-level or
   mutation-site feature vectors.
4. Optionally transform those vectors through an abstraction method such as a
   sparse autoencoder.
5. Infer feature-level selection coefficients from temporal changes in the
   population feature distribution.
6. Convert selection coefficients back into inferred fitness values for each
   sequence.

The central modeling assumption is that a protein variant can be represented by
a feature vector `z`, and that its relative fitness is approximately linear in
that feature vector:

$$
\operatorname{fitness}(z) = \operatorname{baseline} + z^\top s
$$

where `s` is the vector of inferred selection coefficients. Different embedding
types and abstraction methods define different coordinate systems for `z`.

## Inputs and Sequence Reconstruction

The code supports two data modes:

- `CellularDMSInput`: MaveDB-style count tables, optionally with a separate
  functional score CSV.
- `ViralDMSInput`: paired pre-selection and post-selection codon-count files.

Both are converted to a common long-format table:

```text
SequenceIndex, Replicate, Generation, Frequency
```

Despite the column name `Frequency`, the values are initially read as raw counts.
The downstream covariance code normalizes counts within each replicate and
generation, so it treats these values as relative weights at each time point.

For cellular/MaveDB data, `_build_sequence_dataframe_mavedb_primary_keyed`
starts from a reference sequence and variant identifiers such as HGVS nucleotide
strings. The reference can be either nucleotide or protein:

- If the reference is nucleotide, substitutions are applied directly to the
  coding sequence, then translated.
- If the reference is protein, the code infers compatible codon-level effects
  from the amino-acid reference when possible.

The code records:

```text
sequence_to_protein_sequence[SequenceIndex] = full amino-acid sequence
sequence_to_mutation_sites[SequenceIndex] = list of 0-indexed changed residues
```

Stop codons are dropped by default. The wildtype sequence is also added under
`wildtype_key` for cellular data, with an empty mutation-site list.

For viral codon-count data, `build_sequence_dataframe` constructs all observed
single-amino-acid mutants by replacing each wildtype residue with each observed
alternate codon amino acid, skipping synonymous changes and stop-containing
sequences.

## ESM Embeddings

The embedding step uses Hugging Face models from the ESM family:

- ESM-2 models such as `facebook/esm2_t33_650M_UR50D`.
- ESMC models such as `biohub/ESMC-300M`, `biohub/ESMC-600M`, and
  `biohub/ESMC-6B`.

For each protein sequence, `_embed_sequence` tokenizes the sequence with special
tokens enabled, runs the model with `output_hidden_states=True`, removes special
tokens, and returns all residue hidden states from all layers.

The returned array has shape:

$$
(\operatorname{num\_residues},\operatorname{num\_layers},\operatorname{embedding\_dim})
$$

Conceptually, for a sequence of length `L`, transformer layer `ell`, and hidden
dimension `d`, the model produces:

$$
h_i^{(\ell)} \in \mathbb{R}^d
$$

for residue position `i = 1, ..., L`. In array form:

$$
H[\operatorname{residue\_index},
  \operatorname{layer\_index},
  \operatorname{hidden\_dimension}]
$$

The helper `all_residue_representation` is important because it excludes CLS,
EOS, and padding tokens. The code deliberately rejects CLS-token embeddings; all
supported features are derived from real residue representations.

### Layer Selection

`_select_layer` extracts one layer from the full embedding tensor. If the stored
embedding is:

- 1D, it is already a feature vector and is returned unchanged.
- 2D, it is interpreted as `(num_layers, embedding_dim)` and selects one layer.
- 3D, it is interpreted as `(num_residues, num_layers, embedding_dim)` and
  extracts a residue matrix for one layer:

$$
H_{\text{layer}} = H[:, \operatorname{layer\_index}, :]
$$

For raw inference, unpooled 2D per-residue features are not accepted unless the
code explicitly expands them into residue-level pseudo-individuals.

### ESMC Memory-Saving Path

`_embed_sequence_feature_chunks` has a special path for ESMC models that exposes
`model.esmc.embed` and transformer blocks. Instead of materializing all hidden
states for all layers at once, it walks through layers and immediately saves
derived mean, max, and per-residue features for requested layers. This preserves
the same mathematical features while reducing memory pressure.

## Pooling Embeddings

ESM produces one vector per residue per layer. The code implements several ways
to convert those into feature vectors for inference.

### Mean Pooling

For a selected layer, let `h_i` be the hidden state at residue `i`, with sequence
length `L`. Mean pooling computes:

$$
z = \frac{1}{L} \sum_{i=1}^{L} h_i
$$

This produces one `embedding_dim`-length vector per sequence. It represents the
whole sequence as the average residue context. It is simple, stable, and uses
information from all positions, but it can dilute the signal from one or a few
mutated sites.

Implementation:

- `_derive_mean_pool_features`
- `_derive_embedding_type(..., embedding_type="mean_pool")`

### Max Pooling

Max pooling computes an elementwise maximum over residues:

$$
z_j = \max_i h_{i,j}
$$

for each hidden dimension `j`. This also produces one vector per sequence. It
emphasizes the strongest activation of each hidden dimension anywhere in the
sequence. It can preserve localized strong signals, but it discards position and
averaging information.

Implementation:

- `_derive_max_pool_features`
- `_derive_embedding_type(..., embedding_type="max_pool")`

### Per-Residue Mutation-Site Features

For each variant, the code knows which residue positions differ from the
reference. Let that mutation-site set be:

$$
M(\operatorname{seq}) = \{m_1, \ldots, m_q\}
$$

The `per_residue` embedding type selects only those rows:

$$
Z =
\begin{bmatrix}
h_{m_1} \\
h_{m_2} \\
\vdots \\
h_{m_q}
\end{bmatrix}
$$

This gives a matrix with shape:

$$
(\operatorname{number\_of\_mutated\_sites},
 \operatorname{embedding\_dim})
$$

For single mutants, this is one residue vector. For multiple mutants, it is one
vector per mutated site. Wildtype and variants with no amino-acid-changing sites
receive `None` for mutation-site features and are later dropped where needed.

Implementation:

- `_derive_per_residue_features`
- `_derive_embedding_type(..., embedding_type="per_residue")`

### Mutation-Pooled Features

`mutation_pooled` starts from the mutation-site per-residue matrix and averages
over mutated sites:

$$
z = \frac{1}{|M(\operatorname{seq})|}
    \sum_{m \in M(\operatorname{seq})} h_m
$$

This gives one vector per variant, focused only on changed positions. For single
mutants, mutation pooling is exactly the mutated residue embedding. For multiple
mutants, it averages the altered positions and loses which mutation produced
which part of the signal.

Implementation:

- `_pool_per_residue_features`
- `load_embeddings(..., embedding_type="mutation_pooled")`

The code does not save mutation-pooled embeddings directly. It saves
`per_residue` embeddings and derives `mutation_pooled` on load. This avoids
duplicating caches.

### Raw Per-Residue Inference Expansion

Raw `per_residue` embeddings are matrices, but selection inference expects one
vector per row. `_expand_per_residue_features_for_inference` converts each
mutation-site row into a pseudo-individual:

```text
original SequenceIndex = seq
residue feature IDs = seq__site_<mutation_site>
```

The sequence count row is duplicated for each residue feature. This changes the
interpretation: inference is no longer over whole protein variants as single
entities, but over mutation-site feature rows that inherit the same abundance
trajectory as their parent sequence.

## Feature Abstractions

`create_feature_space` optionally transforms raw ESM features into another
coordinate system before inference.

Implemented methods:

- `none`: use raw embedding features.
- `SAE`: sparse autoencoder activations.
- `DeltaSAE`: SAE activations relative to wildtype SAE activations.
- `DeltaEmbSAE`: SAE activations of wildtype-centered raw embeddings.

Declared but not implemented:

- `PCA`: raises `NotImplementedError`.
- `SPCA`: raises `NotImplementedError`.

Before abstraction, all features must be vectors. If `embedding_type` is
`per_residue`, the code expands residue matrices into individual residue vectors
first.

## Sparse Autoencoder Theory

The `SparseAutoencoder` class learns a sparse, overcomplete representation of
ESM-derived vectors. Given an input vector:

$$
x \in \mathbb{R}^d
$$

the model learns `n_features` hidden units, often with `n_features > d`.

### Architecture

The implemented architecture is:

$$
\begin{aligned}
x_{\text{centered}} &= x - b_d \\
z_{\text{raw}} &= W_{\text{enc}} x_{\text{centered}} \\
z &= \operatorname{ReLU}(z_{\text{raw}}) \\
\hat{x} &= W_{\text{dec}} z + b_d
\end{aligned}
$$

where:

- $b_d \in \mathbb{R}^d$ is a learned decoder bias / data center.
- `W_enc` maps input dimensions to hidden features.
- `W_dec` maps sparse hidden features back to input dimensions.
- `z` is the learned sparse feature vector.

Both encoder and decoder linear layers have `bias=False`; all centering is
handled by the shared learned vector `b_d`.

### Why Center With `b_d`?

ESM embeddings have a large common component shared across sequences. If the
autoencoder has to reconstruct this common mean using sparse features, many
hidden units may fire just to represent the background embedding geometry. The
learned bias `b_d` lets the model reconstruct the average input directly:

$$
x \approx b_d + \text{deviations}
$$

The hidden features can then focus on deviations from the mean, which is the
part most relevant for mutation effects and sparse feature discovery.

### Weight Initialization and Decoder Normalization

The decoder is initialized with Kaiming uniform weights. If
`normalize_decoder=True`, decoder columns are projected to unit norm:

$$
\lVert W_{\text{dec}}[:, k] \rVert_2 = 1
$$

The encoder is initialized as the transpose of the decoder:

$$
W_{\text{enc}} = W_{\text{dec}}^\top
$$

During training, decoder columns are renormalized after each optimizer step.
This matters because otherwise the model could reduce an L1 activation penalty
by scaling hidden activations down and decoder weights up while leaving
reconstruction unchanged. Unit-norm decoder columns make activation magnitudes
more comparable across hidden units.

### Normal Sparsity Mode

In `sparsity_mode="normal"`, sparsity comes from an L1 penalty on activations.
For a batch `B`, the loss is:

$$
\mathcal{L}
= \operatorname{MSE}(\hat{x}, x)
  + \lambda \operatorname{mean}(|z|)
$$

where `lambda` is `sparsity_coeff`.

The MSE term forces reconstruction of the original embedding vector. The L1
term encourages most hidden units to be exactly or nearly zero. ReLU already
makes activations nonnegative, so `|z|` is effectively `z`, but the code uses
`z.abs().mean()`.

### TopK Sparsity Mode

In `sparsity_mode="topk"`, the encoder first computes ReLU activations, then
keeps only the top `k` activations within each sample:

$$
z_i = \operatorname{TopK}_k\!\left(
    \operatorname{ReLU}\!\left(W_{\text{enc}}(x_i - b_d)\right)
\right)
$$

All other hidden dimensions are set to zero. The loss is reconstruction MSE
only; the L1 penalty is omitted because the hard TopK mask directly enforces
sparsity.

Interpretation: every sample is represented using at most `k` active dictionary
features.

### BatchTopK Sparsity Mode

In `sparsity_mode="batchtopk"`, the code flattens all activations in the batch
and keeps the largest:

$$
k_{\text{total}} = k \cdot \operatorname{batch\_size}
$$

across the whole batch. This enforces an average of `k` active features per
sample, but it does not require every sample to use exactly `k` features. Some
samples can use more and others fewer.

### Training and Feature Extraction

`_sae_abstraction` trains on a matrix:

$$
X =
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}
\in \mathbb{R}^{n \times d}
$$

using an 80/20 train/test split by default. Hyperparameters include:

- `n_features`, default `2 * input_dim`
- `sparsity_coeff`, default `1e-3`
- `lr`, default `1e-3`
- `epochs`, default `200`
- `batch_size`, default `64`
- `train_frac`, default `0.8`
- `sparsity_mode`, one of `normal`, `topk`, `batchtopk`
- `k`, required for TopK modes

After training, the code encodes every sample:

$$
Z = \operatorname{SAE.encode}(X)
$$

It then computes hidden-unit activation frequencies:

$$
\operatorname{act\_freq}_j
= \frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\!\left[Z_{i,j} > 0\right]
$$

The returned feature space keeps units with `act_freq > 0`. The comment says
"strictly in (0, 1)", but the actual mask is:

$$
\operatorname{active\_mask}_j =
\left(\operatorname{act\_freq}_j > 0\right)
$$

so always-active units are retained. If a checkpoint supplies an `active_mask`,
that mask is reused. If no units are active, the code falls back to the top 10%
by mean activation.

The final SAE feature for each sequence is:

$$
z_{\text{active}}(\operatorname{sequence})
= Z[\operatorname{sequence}, \operatorname{active\_mask}]
$$

### DeltaSAE

`DeltaSAE` trains the SAE on raw embeddings, obtains SAE activations for each
sequence, and subtracts the wildtype activation vector:

$$
z_{\Delta}(\operatorname{seq})
= \operatorname{SAE}(x_{\operatorname{seq}})
  - \operatorname{SAE}(x_{\operatorname{wt}})
$$

This asks: how do the learned sparse features change relative to wildtype in
SAE activation space?

The wildtype row itself is removed from the returned feature dictionary.

### DeltaEmbSAE

`DeltaEmbSAE` first subtracts the wildtype embedding in the original ESM feature
space:

$$
\Delta x(\operatorname{seq}) =
x_{\operatorname{seq}} - x_{\operatorname{wt}}
$$

Then it trains/applies an SAE on those centered delta embeddings:

$$
z_{\Delta}(\operatorname{seq})
= \operatorname{SAE}\!\left(\Delta x(\operatorname{seq})\right)
$$

Optionally, if `subtract_zero_activation=True`, it subtracts the SAE activation
of the zero delta vector:

$$
z_{\Delta}(\operatorname{seq})
= \operatorname{SAE}(x_{\operatorname{seq}} - x_{\operatorname{wt}})
  - \operatorname{SAE}(0)
$$

This differs from `DeltaSAE` because the nonlinear ReLU encoding is applied
after centering in raw embedding space, not before. In general:

$$
\operatorname{SAE}(x_{\operatorname{seq}})
- \operatorname{SAE}(x_{\operatorname{wt}})
\ne
\operatorname{SAE}(x_{\operatorname{seq}} - x_{\operatorname{wt}})
$$

because the SAE encoder includes ReLU, TopK masks, and learned centering.

## Feature Normalization

Before inference, the code can normalize the feature matrix:

$$
\operatorname{features\ shape}
= (n_{\text{sequences}}, n_{\text{features}})
$$

Supported schemes:

- `none`: leave features unchanged.
- `cross_feature`: subtract one global mean and divide by one global standard
  deviation across all entries.
- `per_feature`: standardize each feature dimension separately:

$$
z'_{i,j}
= \frac{z_{i,j} - \operatorname{mean}_i(z_{i,j})}
       {\operatorname{std}_i(z_{i,j}) + 10^{-8}}
$$

Normalization changes the units of the inferred selection coefficients. With
`per_feature`, coefficients are selection effects per one standard deviation of
that feature, rather than per raw embedding unit.

## Selection-Coefficient Inference

The inference theory is implemented in `popDMS.py` by
`compute_dx_covariance_esm`, `mini_infer_esm`, and `infer_gamma_range`.

### Population Feature Mean

At a given replicate `r` and generation/time `t`, suppose variant `a` has count
or frequency `c_a(t)` and feature vector `z_a in R^d`. The code normalizes
within the time point:

$$
w_a(t) = \frac{c_a(t)}{\sum_b c_b(t)}
$$

The population mean feature vector is:

$$
x(t) = \sum_a w_a(t) z_a
$$

This is implemented as:

```text
x[i] = w @ feature_mat
```

where `feature_mat` has one row per sequence and one column per feature.

### Change in Mean Feature Vector

For each replicate, the observed change is:

$$
dx = x(t_{\text{final}}) - x(t_{\text{initial}})
$$

This is the left-hand side of the selection inference problem. It measures how
the population moved through feature space over the selection experiment.

### Population Covariance

The feature covariance at time `t` is:

$$
C(t) =
\mathbb{E}[z z^\top](t)
- \mathbb{E}[z](t)\mathbb{E}[z](t)^\top
$$

where:

$$
\mathbb{E}[z](t) = x(t)
$$

$$
\mathbb{E}[z z^\top](t)
= \sum_a w_a(t) z_a z_a^\top
$$

The code computes the second moment as:

$$
M(t) = (\operatorname{feature\_mat}^\top \odot w)\operatorname{feature\_mat}
$$

and then:

$$
C(t) = M(t) - x(t)x(t)^\top
$$

This is a full feature covariance matrix, not a diagonal approximation.

### Time-Integrated Covariance

The inference uses the time integral of covariance:

$$
I_C = \int C(t)\,dt
$$

The code calls this `icov`, although it is not an inverse covariance; it is the
integrated covariance matrix. It is computed by trapezoidal integration over
the observed generations.

For sorted times:

$$
t_0, t_1, \ldots, t_T
$$

the trapezoid weight for each sampled time is:

$$
\begin{aligned}
\operatorname{weight}_0 &= \frac{t_1 - t_0}{2} \\
\operatorname{weight}_T &= \frac{t_T - t_{T-1}}{2} \\
\operatorname{weight}_i &= \frac{t_{i+1} - t_{i-1}}{2},
\qquad 0 < i < T
\end{aligned}
$$

Then:

$$
I_C = \sum_i \operatorname{weight}_i C(t_i)
$$

Because each `C(t_i)` is positive semidefinite, this weighted sum is also
positive semidefinite when time weights are nonnegative.

### Why `dx = I_C s`?

The inference follows the standard quantitative-genetic / Price-equation form
for directional selection on features. If variant fitness is linear:

$$
\operatorname{fitness}(z)
= \operatorname{baseline} + z^\top s
$$

then selection changes the population mean features approximately according to:

$$
\frac{dx}{dt} = C(t)s
$$

where `C(t)` is the population covariance of features. Intuitively, a feature
can only respond to selection if it varies in the population, and correlated
features respond together according to their covariance.

Integrating over time gives:

$$
\Delta x = \left(\int C(t)\,dt\right)s
$$

or:

$$
dx = I_C s
$$

Thus selection inference is a linear inverse problem: find `s` such that the
observed change in population mean features is explained by the integrated
feature covariance.

### Regularized Solve

Directly solving:

$$
I_C s = dx
$$

can be unstable because `I_C` may be singular or ill-conditioned, especially
when the feature dimension is large relative to the number of observed variants.
The code uses ridge regularization:

$$
(I_C + \gamma I)s = dx
$$

so:

$$
s = (I_C + \gamma I)^{-1}dx
$$

`gamma` controls shrinkage:

- Small `gamma`: higher-variance, less biased estimates that follow the data
  closely.
- Large `gamma`: lower-variance, more strongly shrunk estimates.

### Eigendecomposition Implementation

For numerical efficiency, `_prepare_eigendecomp` diagonalizes each replicate's
integrated covariance:

$$
I_C = V\operatorname{diag}(\lambda)V^\top
$$

Then:

$$
s = V\left(\frac{V^\top dx}{\lambda + \gamma}\right)
$$

This avoids repeatedly inverting large matrices during gamma sweeps.

### Choosing `gamma`

If the user supplies `gamma`, it is used directly. If there is only one
replicate, `mini_infer_esm` defaults to:

$$
\gamma = 1
$$

If multiple replicates are available, the code sweeps:

$$
\gamma_{\text{values}}
= \operatorname{logspace}\!\left(
    \log_{10}\!\left(\frac{1}{\operatorname{max\_reads}}\right),
    4,
    20
  \right)
$$

with `max_reads=1e2` by default, so the default range is:

$$
0.01 \le \gamma \le 10000
$$

For each gamma, it computes replicate-specific selection vectors and measures
the mean pairwise Pearson correlation between replicates. `get_best_regularization`
then selects a gamma based on the shape of this correlation curve. In broad
terms, the selected gamma is intended to be strong enough to improve
cross-replicate consistency without simply choosing the most over-regularized
solution.

`plot_regularization_curve` exposes this sweep for inspection.

### Per-Replicate and Joint Selection Coefficients

For each replicate `r`, the code computes:

$$
s_r = (I_{C,r} + \gamma I)^{-1}dx_r
$$

It also computes a joint estimate by summing information across replicates:

$$
dx_{\text{joint}} = \sum_r dx_r
$$

$$
I_{C,\text{joint}} = \sum_r I_{C,r}
$$

and solving:

$$
s_{\text{joint}}
= \left(I_{C,\text{joint}}
  + n_{\text{replicates}}\gamma I\right)^{-1}
  dx_{\text{joint}}
$$

The joint regularization uses `n_replicates * gamma`, matching the summed scale
of the covariance matrices.

### Error Bars

When `calc_error_bars=True`, the code estimates coefficient uncertainty from
the diagonal of the regularized inverse:

$$
\operatorname{error\_bar}_j
= \sqrt{\left[(I_C + \gamma I)^{-1}\right]_{j,j}}
$$

Using the eigendecomposition:

$$
(I_C + \gamma I)^{-1}
= V\operatorname{diag}\!\left(\frac{1}{\lambda + \gamma}\right)V^\top
$$

so the diagonal is:

$$
\operatorname{diag}_j
= \sum_k \frac{V_{j,k}^2}{\lambda_k + \gamma}
$$

The code computes:

$$
\sqrt{\operatorname{eig\_vec}^{\,2}\operatorname{@}\operatorname{inv\_denom}}
$$

These are not currently requested by `esmDMS.run_feature_inference`, because it
calls `mini_infer_esm` with default `calc_error_bars=False`.

## Inferred Fitness

After selection coefficients are inferred, a sequence's predicted fitness is
computed by a dot product:

$$
\operatorname{fitness}(\operatorname{seq})
= \operatorname{baseline} + z_{\operatorname{seq}}^\top s
$$

`fitness_dataframe` uses `baseline=1.0` by default and uses `s_joint` unless
`use_joint=False`.

For replicate-specific fitness:

$$
\operatorname{fitness}_r(\operatorname{seq})
= \operatorname{baseline} + z_{\operatorname{seq}}^\top s_r
$$

For method-comparison helpers such as `_fitness_for_method`, the code often
uses:

$$
\operatorname{fitness}(\operatorname{seq})
= z_{\operatorname{seq}}^\top s_{\text{joint}}
$$

without adding the baseline, because rank/correlation comparisons are
insensitive to a constant offset.

For `DeltaSAE` and `DeltaEmbSAE`, features are already relative to wildtype, so
the dot product is interpretable as a mutant effect relative to the chosen
wildtype feature baseline. `fitness_dataframe` still adds the numeric baseline.

## Functional Score Comparisons

If a MaveDB score CSV is supplied, `plot_functional_score_comparison` merges
inferred fitness with a score column such as `score` and computes Spearman rank
correlation:

$$
\rho =
\operatorname{Spearman}(
  \operatorname{inferred\ fitness},
  \operatorname{functional\ score}
)
$$

The plot is a diagnostic: it tests whether the inferred dynamic-selection
fitness scores agree with externally provided functional measurements. It does
not affect inference.

## Replicate Consistency Diagnostics

The plotting helpers compare either:

- selection coefficients `s_r` across replicates, or
- inferred sequence fitness `features @ s_r` across replicates.

Selection-coefficient comparisons ask whether the inferred feature-level model
is reproducible. Fitness comparisons ask whether predicted variant rankings or
effects are reproducible after projecting coefficients back onto sequence
features.

The code reports both Pearson and Spearman correlations in scatter-grid plots:

- Pearson: linear agreement in coefficient/fitness magnitudes.
- Spearman: rank-order agreement.

## Batch Jobs and Caching

Large ESM models and large DMS libraries are expensive to process, so the code
has disk-backed caching and Slurm job creation helpers.

Embedding caches are keyed by:

```text
dataset_name, embedding_model, embedding_type, abstraction_method, layer
```

Important cache behavior:

- Mean, max, and per-residue features can be saved per layer.
- Mutation-pooled features are derived from saved per-residue features.
- SAE models and visualization data are saved separately under `sae_models`.
- Inference results are saved by layer, embedding type, abstraction method, and
  normalization scheme.

The batch embedding path splits sequence IDs into chunks, embeds each chunk,
and then merges chunk files. For ESMC, the chunk worker saves already-pooled
features rather than all hidden states when possible.

## Important Interpretation Notes

1. `Frequency` is normalized within each replicate/time point during inference.
   Absolute count scale affects neither `x(t)` nor `C(t)` except through zero
   time points and which variants are present.

2. The inferred selection coefficients live in the chosen feature coordinate
   system. Raw ESM, mutation-pooled ESM, SAE, DeltaSAE, and DeltaEmbSAE
   coefficients are not directly interchangeable unless transformed into the
   same feature basis.

3. Pooling controls the biological object being modeled. Mean/max pooling model
   whole-sequence representation shifts; mutation pooling focuses on altered
   residues; raw per-residue expansion treats mutation-site rows as the
   inference units.

4. The covariance matrix determines which directions in feature space are
   learnable. A feature direction with little variation in the population cannot
   be estimated reliably, regardless of how biologically meaningful it might be.

5. Ridge regularization is not just a numerical detail. It defines the bias and
   variance of `s`, and the default automatic choice uses replicate agreement as
   its criterion.

6. Sparse autoencoders do not produce selection coefficients themselves. They
   define a new sparse feature basis; the same population-genetic inference is
   then run on that basis.

7. PCA and SPCA are present in the public method vocabulary but are not
   implemented in this file. Calling them raises `NotImplementedError`.

8. The SAE active-neuron selection currently keeps all neurons that ever fire,
   not only neurons with activation frequency strictly between zero and one.
   The nearby comment is stricter than the implemented mask.

9. Wildtype handling differs by feature type. Mutation-site embeddings assign
   wildtype no mutation-site feature, so it is dropped for those analyses.
   Delta methods require a wildtype key so that mutant features can be expressed
   relative to wildtype.

## Compact End-to-End Formula Summary

For a sequence `a`, choose a feature extraction map:

$$
z_a = \phi(\operatorname{ESM}(\operatorname{sequence}_a))
$$

At replicate `r`, time `t`, normalize observed counts:

$$
w_a(t) = \frac{c_a(t)}{\sum_b c_b(t)}
$$

Compute the population feature mean:

$$
x_r(t) = \sum_a w_a(t)z_a
$$

Compute the feature covariance:

$$
C_r(t)
= \sum_a w_a(t)z_a z_a^\top
  - x_r(t)x_r(t)^\top
$$

Integrate covariance over time:

$$
I_{C,r} = \int C_r(t)\,dt
$$

Measure feature displacement:

$$
dx_r = x_r(t_{\text{final}}) - x_r(t_{\text{initial}})
$$

Infer replicate selection coefficients:

$$
s_r = (I_{C,r} + \gamma I)^{-1}dx_r
$$

Infer joint coefficients:

$$
s_{\text{joint}}
= \left(\sum_r I_{C,r}
  + n_{\text{replicates}}\gamma I\right)^{-1}
  \sum_r dx_r
$$

Predict fitness:

$$
\operatorname{fitness}_a
= \operatorname{baseline} + z_a^\top s
$$

Everything in `esmDMS.py` is organized around choosing `phi`, caching the
resulting `z_a`, and running this inference pipeline reproducibly across layers,
embedding types, abstraction methods, and replicates.
