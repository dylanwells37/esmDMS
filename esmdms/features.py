"""Protein language-model features, masked-marginal priors, and SAE features."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import torch

from .schema import Dataset, FeatureArtifact, SEQUENCE_ID


Pooling = Literal["mean", "max"]
SparsityMode = Literal["normal", "topk", "batchtopk"]
AA_ALPHABET = tuple("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_WINDOW_SIZE = 2048
ProgressCallback = Callable[[str, int, int], None]


def llr_sequence_ids(dataset: Dataset) -> tuple[str, ...]:
    """Return canonical rows eligible for masked-marginal amino-acid LLRs."""
    return tuple(_eligible_llr_variants(dataset)[SEQUENCE_ID].astype(str))


def _eligible_llr_variants(dataset: Dataset):
    return dataset.variants[
        ~dataset.variants["is_synonymous"]
        & dataset.variants["mutant_aa"].astype(str).isin(AA_ALPHABET)
    ]


def _llr_positions(dataset: Dataset) -> tuple[int, ...]:
    return tuple(
        dict.fromkeys(_eligible_llr_variants(dataset)["position"].astype(int))
    )


def _torch_dtype(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    try:
        return {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported torch dtype {name!r}.") from error


def load_language_model(
    model_name: str,
    *,
    device: str | None = None,
    dtype: str | None = None,
):
    """Load a Hugging Face ESM-C safetensors checkpoint into the ESM model."""
    from esm.models.esmc import ESMC
    from esm.tokenization import get_esmc_model_tokenizers
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    resolved_dtype = _torch_dtype(dtype)
    snapshot = Path(
        snapshot_download(
            repo_id=model_name,
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "model.safetensors.index.json",
                "model-*.safetensors",
            ],
        )
    )
    config = json.loads((snapshot / "config.json").read_text())
    if config.get("model_type") != "esmc":
        raise ValueError(f"Model {model_name!r} is not an ESM-C checkpoint.")

    tokenizer = get_esmc_model_tokenizers()
    model = ESMC(
        d_model=int(config["d_model"]),
        n_heads=int(config["n_heads"]),
        n_layers=int(config["n_layers"]),
        tokenizer=tokenizer,
        use_flash_attn=False,
    )
    state = {}
    for checkpoint_file in _checkpoint_files(snapshot):
        for name, value in load_file(checkpoint_file).items():
            if (converted := _esmc_state_name(name)) is not None:
                if converted in state:
                    raise ValueError(
                        f"Duplicate tensor {converted!r} in model checkpoint."
                    )
                state[converted] = value
    model.load_state_dict(state, strict=True, assign=True)
    if resolved_dtype is None:
        model = model.to(target)
    else:
        model = model.to(device=target, dtype=resolved_dtype)
    model.eval()
    return tokenizer, model


def _checkpoint_files(snapshot: Path) -> tuple[Path, ...]:
    """Resolve a single-file or Hugging Face sharded safetensors checkpoint."""
    single_file = snapshot / "model.safetensors"
    if single_file.is_file():
        return (single_file,)
    index_file = snapshot / "model.safetensors.index.json"
    if not index_file.is_file():
        raise FileNotFoundError(
            f"No model.safetensors or model.safetensors.index.json in {snapshot}"
        )
    index = json.loads(index_file.read_text())
    filenames = tuple(dict.fromkeys(index.get("weight_map", {}).values()))
    if not filenames:
        raise ValueError(f"Checkpoint index {index_file} has an empty weight_map.")
    files = tuple(snapshot / str(filename) for filename in filenames)
    missing = [path.name for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint index references missing shards: {', '.join(missing)}"
        )
    return files


def _esmc_state_name(name: str) -> str | None:
    """Map Hugging Face ESM-C module names to the official ESM implementation."""
    if name.endswith("._extra_state"):
        return None
    if name.startswith("esmc."):
        name = name.removeprefix("esmc.")
    elif name.startswith("lm_head."):
        name = "sequence_head." + name.removeprefix("lm_head.")
    else:
        return None
    replacements = {
        ".attn.layernorm_qkv.layer_norm_weight": ".attn.layernorm_qkv.0.weight",
        ".attn.layernorm_qkv.layer_norm_bias": ".attn.layernorm_qkv.0.bias",
        ".attn.layernorm_qkv.weight": ".attn.layernorm_qkv.1.weight",
        ".ffn.layer_norm_weight": ".ffn.0.weight",
        ".ffn.layer_norm_bias": ".ffn.0.bias",
        ".ffn.fc1_weight": ".ffn.1.weight",
        ".ffn.fc2_weight": ".ffn.3.weight",
    }
    for source, destination in replacements.items():
        name = name.replace(source, destination)
    return name


def _model_device(model) -> torch.device:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def analysis_window(
    dataset: Dataset,
    window_size: int | None,
    truncate: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Return one fixed zero-based reference window covering all substitutions."""
    truncate = dataset.truncation if truncate is None else truncate
    length = len(dataset.reference_sequence)
    if window_size is not None and window_size <= 0:
        raise ValueError("window_size must be positive or None.")
    substitutions = dataset.variants.loc[
        ~dataset.variants["is_synonymous"], "position"
    ].astype(int)
    if truncate is not None:
        start, end = (int(value) for value in truncate)
        if start < 1 or end < start or end > length:
            raise ValueError(
                f"truncate must be a valid 1-based inclusive interval within "
                f"the {length}-residue reference sequence."
            )
        if window_size is not None and end - start + 1 > window_size:
            raise ValueError(
                f"Truncation interval spans {end - start + 1} residues, "
                f"exceeding window_size={window_size}."
            )
        if not substitutions.empty and (
            int(substitutions.min()) < start or int(substitutions.max()) > end
        ):
            raise ValueError(
                f"Truncation interval {start}-{end} does not cover every "
                "assayed substitution."
            )
        return start - 1, end
    if window_size is None or length <= window_size:
        return 0, length
    if substitutions.empty:
        raise ValueError("Cannot place an analysis window without substitutions.")
    first = int(substitutions.min()) - 1
    last = int(substitutions.max()) - 1
    if last - first + 1 > window_size:
        raise ValueError(
            f"Assayed substitutions span {last - first + 1} residues, exceeding "
            f"window_size={window_size}."
        )
    center = (first + last) // 2
    start = max(0, min(center - window_size // 2, length - window_size))
    end = start + window_size
    if not (start <= first <= last < end):
        raise RuntimeError("Could not place a fixed window around all substitutions.")
    return start, end


def _tokenize(sequence: str, tokenizer) -> dict:
    """Tokenize one sequence and guarantee an explicit attention mask.

    The mask is materialized here rather than defaulted at each use site, so
    every downstream reader sees the same tensor.
    """
    inputs = dict(
        tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True,
            return_special_tokens_mask=True,
        )
    )
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    return inputs


def _residue_hidden_states(
    sequence: str,
    tokenizer,
    model,
    layers: Sequence[int] | None,
) -> tuple[dict[int, np.ndarray], int]:
    inputs = _tokenize(sequence, tokenizer)
    special_mask = inputs.pop("special_tokens_mask").bool()
    residue_mask = (inputs["attention_mask"].bool() & ~special_mask).squeeze(0)
    model_inputs = {
        key: value.to(_model_device(model)) for key, value in inputs.items()
    }
    with torch.inference_mode():
        outputs = model(
            sequence_tokens=model_inputs["input_ids"],
            sequence_id=model_inputs["attention_mask"].bool(),
        )
    n_layers = int(outputs.hidden_states.shape[0])
    selected_layers = (
        tuple(range(n_layers + 1))
        if layers is None
        else tuple(int(layer) for layer in layers)
    )
    invalid = [layer for layer in selected_layers if not 0 <= layer <= n_layers]
    if invalid:
        raise ValueError(
            f"Invalid layer indices {invalid}; model exposes layers 0 through "
            f"{n_layers}."
        )
    layer_values = {0: model.embed(model_inputs["input_ids"])}
    layer_values.update(
        {
            layer: (
                outputs.embeddings
                if layer == n_layers
                else outputs.hidden_states[layer - 1]
            )
            for layer in selected_layers
            if layer > 0
        }
    )
    states = {
        layer: layer_values[layer][0, residue_mask.to(layer_values[layer].device), :]
        .detach()
        .float()
        .cpu()
        .numpy()
        for layer in selected_layers
    }
    if any(state.shape[0] != len(sequence) for state in states.values()):
        raise ValueError(
            "Tokenizer residue count does not match the protein sequence length."
        )
    return states, n_layers


def embed(
    dataset: Dataset,
    model_name: str,
    *,
    layers: Sequence[int] | None = None,
    pooling: Pooling = "max",
    window_size: int | None = DEFAULT_WINDOW_SIZE,
    truncate: tuple[int, int] | None = None,
    device: str | None = None,
    dtype: str | None = None,
) -> dict[int, FeatureArtifact]:
    """Embed each unique protein sequence and return one artifact per layer."""
    tokenizer, model = load_language_model(model_name, device=device, dtype=dtype)
    return _embed_with_model(
        dataset,
        model_name,
        tokenizer,
        model,
        sequence_ids=dataset.sequence_ids,
        layers=layers,
        pooling=pooling,
        window_size=window_size,
        truncate=truncate,
    )


def _embed_with_model(
    dataset: Dataset,
    model_name: str,
    tokenizer,
    model,
    *,
    sequence_ids: Sequence[str],
    layers: Sequence[int] | None,
    pooling: Pooling,
    window_size: int | None,
    truncate: tuple[int, int] | None,
    provenance_extra: dict | None = None,
    progress: ProgressCallback | None = None,
) -> dict[int, FeatureArtifact]:
    if pooling not in {"mean", "max"}:
        raise ValueError("pooling must be 'mean' or 'max'.")
    sequence_ids = tuple(str(value) for value in sequence_ids)
    if not sequence_ids:
        raise ValueError("Embedding shard contains no sequence ids.")
    unknown = sorted(set(sequence_ids).difference(dataset.sequence_ids))
    if unknown:
        raise ValueError(f"Unknown embedding sequence ids: {', '.join(unknown[:5])}")

    window_start, window_end = analysis_window(dataset, window_size, truncate)
    full_sequence_by_id = dict(
        zip(
            dataset.variants[SEQUENCE_ID].astype(str),
            dataset.variants["protein_sequence"].astype(str),
        )
    )
    sequence_by_id = {
        sequence_id: full_sequence_by_id[sequence_id][window_start:window_end]
        for sequence_id in sequence_ids
    }
    reducer = np.mean if pooling == "mean" else np.max
    pooled_by_layer: dict[int, dict[str, np.ndarray]] = {}
    selected_layers: tuple[int, ...] | None = None
    final_layer: int | None = None
    unique_sequences = tuple(dict.fromkeys(sequence_by_id.values()))
    for completed, sequence in enumerate(unique_sequences, start=1):
        states, final_layer = _residue_hidden_states(sequence, tokenizer, model, layers)
        if selected_layers is None:
            selected_layers = tuple(states)
            pooled_by_layer = {layer: {} for layer in selected_layers}
        for layer in selected_layers:
            pooled_by_layer[layer][sequence] = reducer(states[layer], axis=0).astype(
                np.float32
            )
        if progress is not None:
            progress("embedding", completed, len(unique_sequences))

    if selected_layers is None:
        raise ValueError("Embedding shard contains no protein sequences.")
    artifacts = {}
    for layer in selected_layers:
        values = np.vstack(
            [
                pooled_by_layer[layer][sequence_by_id[sequence_id]]
                for sequence_id in sequence_ids
            ]
        )
        provenance = {
            "model": model_name,
            "layer": layer,
            # The final index returns the model's post-LayerNorm output state;
            # every earlier index is a raw transformer block output. Layers are
            # therefore on different scales and must not be compared naively.
            "final_layer_norm_applied": layer == final_layer,
            "pooling": pooling,
            "window_selection": (
                "custom"
                if truncate is not None
                else "dataset" if dataset.truncation is not None else "automatic"
            ),
            "window_start": window_start + 1,
            "window_end": window_end,
            "window_size": window_end - window_start,
        }
        provenance.update(provenance_extra or {})
        artifacts[layer] = FeatureArtifact(
            sequence_ids,
            values,
            tuple(f"embedding_{index:05d}" for index in range(values.shape[1])),
            "embedding",
            dataset.name,
            provenance,
        )
    return artifacts


def _amino_acid_token_id(tokenizer, amino_acid: str) -> int | None:
    token_ids = tokenizer(amino_acid, add_special_tokens=False).get("input_ids", [])
    return int(token_ids[0]) if len(token_ids) == 1 else None


def masked_marginal_llr(
    dataset: Dataset,
    model_name: str,
    *,
    window_size: int | None = DEFAULT_WINDOW_SIZE,
    truncate: tuple[int, int] | None = None,
    device: str | None = None,
    dtype: str | None = None,
) -> FeatureArtifact:
    """Compute mutant-minus-wildtype masked marginal log likelihood ratios."""
    tokenizer, model = load_language_model(model_name, device=device, dtype=dtype)
    positions = _llr_positions(dataset)
    return _llr_with_model(
        dataset,
        model_name,
        tokenizer,
        model,
        positions=positions,
        window_size=window_size,
        truncate=truncate,
    )


def _llr_with_model(
    dataset: Dataset,
    model_name: str,
    tokenizer,
    model,
    *,
    positions: Sequence[int],
    window_size: int | None,
    truncate: tuple[int, int] | None,
    provenance_extra: dict | None = None,
    progress: ProgressCallback | None = None,
) -> FeatureArtifact:
    if tokenizer.mask_token_id is None:
        raise ValueError(f"Tokenizer for {model_name!r} has no mask token.")

    positions = tuple(int(position) for position in positions)
    if not positions:
        raise ValueError("LLR shard contains no substitution positions.")
    eligible = _eligible_llr_variants(dataset)
    eligible = eligible[eligible["position"].astype(int).isin(positions)].copy()
    if eligible.empty:
        raise ValueError("LLR shard contains no eligible amino-acid substitutions.")

    window_start, window_end = analysis_window(dataset, window_size, truncate)
    reference_window = dataset.reference_sequence[window_start:window_end]

    inputs = _tokenize(reference_window, tokenizer)
    special_mask = inputs.pop("special_tokens_mask").bool().squeeze(0)
    attention = inputs["attention_mask"].bool().squeeze(0)
    residue_positions = torch.where(attention & ~special_mask)[0]
    if len(residue_positions) != len(reference_window):
        raise ValueError(
            "Tokenizer residue count does not match the reference sequence length."
        )
    inputs = {key: value.to(_model_device(model)) for key, value in inputs.items()}
    token_ids = {aa: _amino_acid_token_id(tokenizer, aa) for aa in AA_ALPHABET}

    llr_by_change: dict[tuple[int, str], float] = {}
    grouped = tuple(eligible.groupby("position", sort=True))
    for completed, (position, group) in enumerate(grouped, start=1):
        position = int(position)
        wt_aa = str(group["wt_aa"].iloc[0])
        wt_id = token_ids.get(wt_aa)
        if wt_id is None:
            raise ValueError(
                f"Could not resolve a tokenizer id for wildtype amino acid {wt_aa!r}."
            )
        masked = {key: value.clone() for key, value in inputs.items()}
        token_position = int(residue_positions[position - window_start - 1])
        masked["input_ids"][0, token_position] = tokenizer.mask_token_id
        with torch.inference_mode():
            outputs = model(
                sequence_tokens=masked["input_ids"],
                sequence_id=masked["attention_mask"].bool(),
            )
            logits = outputs.sequence_logits[0, token_position].float()
            log_probabilities = torch.log_softmax(logits, dim=-1)
        for mutant_aa in group["mutant_aa"].astype(str).unique():
            mutant_id = token_ids.get(mutant_aa)
            if mutant_id is None:
                raise ValueError(
                    f"Could not resolve a tokenizer id for amino acid {mutant_aa!r}."
                )
            llr_by_change[(position, mutant_aa)] = float(
                log_probabilities[mutant_id] - log_probabilities[wt_id]
            )
        if progress is not None:
            progress("llr", completed, len(grouped))

    values = np.asarray(
        [
            llr_by_change[(int(row.position), str(row.mutant_aa))]
            for row in eligible.itertuples()
        ],
        dtype=np.float32,
    )[:, None]
    provenance = {
        "model": model_name,
        "definition": "log_p_mutant_minus_log_p_wildtype",
        "orientation": "raw_llr",
        "window_selection": (
            "custom"
            if truncate is not None
            else "dataset" if dataset.truncation is not None else "automatic"
        ),
        "window_start": window_start + 1,
        "window_end": window_end,
        "window_size": window_end - window_start,
    }
    provenance.update(provenance_extra or {})
    return FeatureArtifact(
        tuple(eligible[SEQUENCE_ID].astype(str)),
        values,
        ("llr",),
        "llr_prior",
        dataset.name,
        provenance,
    )


def embed_and_llr(
    dataset: Dataset,
    model_name: str,
    *,
    layers: Sequence[int] | None = None,
    pooling: Pooling = "max",
    window_size: int | None = DEFAULT_WINDOW_SIZE,
    truncate: tuple[int, int] | None = None,
    device: str | None = None,
    dtype: str | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    progress: ProgressCallback | None = None,
) -> tuple[dict[int, FeatureArtifact], FeatureArtifact]:
    """Generate a deterministic embedding/LLR shard with one model load."""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive.")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be between zero and num_shards - 1.")

    sequence_ids = dataset.sequence_ids[shard_index::num_shards]
    positions = _llr_positions(dataset)[shard_index::num_shards]
    if not sequence_ids:
        raise ValueError(
            f"Embedding shard {shard_index} is empty; reduce num_shards={num_shards}."
        )
    if not positions:
        raise ValueError(
            f"LLR shard {shard_index} is empty; reduce num_shards={num_shards}."
        )

    tokenizer, model = load_language_model(model_name, device=device, dtype=dtype)
    shard_provenance = {
        "shard_index": shard_index,
        "num_shards": num_shards,
    }
    embeddings = _embed_with_model(
        dataset,
        model_name,
        tokenizer,
        model,
        sequence_ids=sequence_ids,
        layers=layers,
        pooling=pooling,
        window_size=window_size,
        truncate=truncate,
        provenance_extra=shard_provenance,
        progress=progress,
    )
    llr = _llr_with_model(
        dataset,
        model_name,
        tokenizer,
        model,
        positions=positions,
        window_size=window_size,
        truncate=truncate,
        provenance_extra=shard_provenance,
        progress=progress,
    )
    return embeddings, llr


class SparseAutoencoder(torch.nn.Module):
    """Sparse autoencoder with a tied input/output centering bias."""

    def __init__(
        self,
        input_dim: int,
        n_features: int,
        mode: SparsityMode = "normal",
        k: int | None = None,
    ):
        super().__init__()
        if mode not in {"normal", "topk", "batchtopk"}:
            raise ValueError(f"Unsupported sparsity mode {mode!r}.")
        if mode in {"topk", "batchtopk"} and (k is None or k <= 0):
            raise ValueError(f"{mode} requires a positive k.")
        self.encoder = torch.nn.Linear(input_dim, n_features, bias=False)
        self.decoder = torch.nn.Linear(n_features, input_dim, bias=False)
        self.center = torch.nn.Parameter(torch.zeros(input_dim))
        self.mode = mode
        self.k = k
        # BatchTopK couples samples through a per-batch ranking. After training,
        # a fixed threshold replaces that ranking so a sequence's activations no
        # longer depend on which other sequences are encoded alongside it.
        self.register_buffer("activation_threshold", torch.tensor(float("nan")))
        torch.nn.init.kaiming_uniform_(self.decoder.weight)
        self.normalize_decoder()
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder.weight.T)

    def normalize_decoder(self) -> None:
        with torch.no_grad():
            self.decoder.weight.div_(
                self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
            )

    def _batch_topk(self, activations: torch.Tensor) -> torch.Tensor:
        flat = activations.flatten()
        count = min(int(self.k) * activations.shape[0], flat.numel())
        indices = flat.topk(count).indices
        return (flat * torch.zeros_like(flat).scatter_(0, indices, 1.0)).view_as(
            activations
        )

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        activations = torch.relu(self.encoder(values - self.center))
        if self.mode == "topk":
            k = min(int(self.k), activations.shape[1])
            indices = activations.topk(k, dim=1).indices
            mask = torch.zeros_like(activations).scatter_(1, indices, 1.0)
            activations = activations * mask
        elif self.mode == "batchtopk":
            threshold = float(self.activation_threshold)
            if self.training or not np.isfinite(threshold):
                activations = self._batch_topk(activations)
            else:
                activations = activations * (activations > threshold)
        return activations

    @torch.no_grad()
    def calibrate_threshold(
        self, values: torch.Tensor, batch_size: int
    ) -> float | None:
        """Fix the BatchTopK decision threshold from the training distribution.

        The threshold is the mean smallest surviving activation across training
        batches, the standard BatchTopK hand-off from a batch ranking to a
        sample-independent rule.
        """
        if self.mode != "batchtopk" or len(values) == 0:
            return None
        was_training = self.training
        self.train()
        minima = []
        for start in range(0, len(values), batch_size):
            activations = self.encode(values[start : start + batch_size])
            surviving = activations[activations > 0]
            if surviving.numel():
                minima.append(float(surviving.min()))
        self.train(was_training)
        if not minima:
            return None
        threshold = float(np.mean(minima))
        self.activation_threshold.fill_(threshold)
        return threshold

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        activations = self.encode(values)
        return self.decoder(activations) + self.center, activations


@dataclass(frozen=True)
class SAEConfig:
    n_features: int | None = None
    sparsity: float = 1e-3
    learning_rate: float = 1e-3
    epochs: int = 200
    batch_size: int = 64
    train_fraction: float = 0.8
    mode: SparsityMode = "normal"
    k: int | None = None
    seed: int = 42
    active_frequency: float = 1e-3
    center_on_reference: bool = True


def train_sae(
    dataset: Dataset,
    embeddings: FeatureArtifact,
    *,
    config: SAEConfig = SAEConfig(),
    model_path: str | Path | None = None,
    device: str | None = None,
) -> FeatureArtifact:
    """Train an SAE and return active features in the canonical artifact format."""
    if embeddings.kind != "embedding":
        raise ValueError("SAE input must be an embedding artifact.")
    if embeddings.dataset != dataset.name:
        raise ValueError("Embedding and dataset names do not match.")
    values = embeddings.align(dataset.sequence_ids).astype(np.float32)
    if config.center_on_reference:
        reference_rows = (
            dataset.variants["protein_sequence"]
            .astype(str)
            .eq(dataset.reference_sequence)
            .to_numpy()
        )
        if not reference_rows.any():
            raise ValueError(
                "center_on_reference requires a reference protein row in variants.csv."
            )
        values = values - values[np.flatnonzero(reference_rows)[0]]

    protein_sequences = dataset.variants["protein_sequence"].astype(str).to_numpy()
    unique_indices = np.unique(protein_sequences, return_index=True)[1]
    fit_values = values[np.sort(unique_indices)]
    if len(fit_values) < 2:
        raise ValueError("SAE training requires at least two unique protein sequences.")
    n_features = config.n_features or values.shape[1] * 2
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    generator = np.random.default_rng(config.seed)
    split = max(
        1, min(len(fit_values) - 1, int(len(fit_values) * config.train_fraction))
    )
    order = generator.permutation(len(fit_values))
    train = torch.from_numpy(fit_values[order[:split]]).to(target)
    # train_fraction holds sequences back, so they are scored rather than
    # silently discarded; the held-out error is what makes the split meaningful.
    validation = torch.from_numpy(fit_values[order[split:]]).to(target)

    torch.manual_seed(config.seed)
    model = SparseAutoencoder(values.shape[1], n_features, config.mode, config.k).to(
        target
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for _ in range(config.epochs):
        order_t = torch.randperm(len(train), device=target)
        for start in range(0, len(train), config.batch_size):
            batch = train[order_t[start : start + config.batch_size]]
            optimizer.zero_grad()
            reconstruction, activations = model(batch)
            loss = torch.nn.functional.mse_loss(reconstruction, batch)
            if config.mode == "normal":
                loss = loss + config.sparsity * activations.abs().mean()
            loss.backward()
            optimizer.step()
            model.normalize_decoder()

    threshold = model.calibrate_threshold(train, config.batch_size)

    all_values = torch.from_numpy(values).to(target)
    model.eval()
    with torch.inference_mode():
        reconstruction, activations = model(all_values)
        train_error = float(
            torch.nn.functional.mse_loss(model(train)[0], train).detach().cpu()
        )
        validation_error = (
            float(
                torch.nn.functional.mse_loss(model(validation)[0], validation)
                .detach()
                .cpu()
            )
            if len(validation)
            else float("nan")
        )
    activations_np = activations.float().cpu().numpy()
    activation_frequency = (activations_np > 0).mean(axis=0)
    active = (activation_frequency >= config.active_frequency) & (
        activation_frequency <= 1.0 - config.active_frequency
    )
    if not active.any():
        raise RuntimeError("SAE produced no non-degenerate active features.")

    reconstruction_error = float(
        torch.nn.functional.mse_loss(reconstruction, all_values).detach().cpu()
    )
    provenance = {
        "source": embeddings.provenance,
        "config": asdict(config),
        "input_features": values.shape[1],
        "active_features": int(active.sum()),
        "reconstruction_mse": reconstruction_error,
        "train_sequences": int(len(train)),
        "validation_sequences": int(len(validation)),
        "train_reconstruction_mse": train_error,
        "validation_reconstruction_mse": validation_error,
        "activation_threshold": threshold,
    }
    if model_path is not None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                # A tensor rather than a NumPy array so the checkpoint stays
                # loadable under torch.load(weights_only=True).
                "active_mask": torch.from_numpy(active),
                "input_dim": values.shape[1],
                "n_features": n_features,
                "provenance": provenance,
            },
            model_path,
        )
        provenance["model_path"] = str(model_path)

    return FeatureArtifact(
        dataset.sequence_ids,
        activations_np[:, active],
        tuple(f"sae_{index:05d}" for index in np.flatnonzero(active)),
        "sae",
        dataset.name,
        provenance,
    )
