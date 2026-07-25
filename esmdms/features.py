"""Protein language-model masked-marginal LLR calculations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from .schema import Dataset, FeatureArtifact, SEQUENCE_ID


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
    shard_index: int = 0,
    num_shards: int = 1,
    progress: ProgressCallback | None = None,
) -> FeatureArtifact:
    """Compute mutant-minus-wildtype masked marginal log likelihood ratios."""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive.")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be between zero and num_shards - 1.")
    positions = _llr_positions(dataset)[shard_index::num_shards]
    if not positions:
        raise ValueError(
            f"LLR shard {shard_index} is empty; reduce num_shards={num_shards}."
        )
    tokenizer, model = load_language_model(model_name, device=device, dtype=dtype)
    return _llr_with_model(
        dataset,
        model_name,
        tokenizer,
        model,
        positions=positions,
        window_size=window_size,
        truncate=truncate,
        provenance_extra={"shard_index": shard_index, "num_shards": num_shards},
        progress=progress,
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
