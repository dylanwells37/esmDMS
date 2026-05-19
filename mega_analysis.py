"""
# Simulation-based analyses (fitness, sel_coeffs)
python mega_analysis.py Ube4b plots/ --sim_config configs/simulation_config.json --fitness
python mega_analysis.py Ube4b plots/ --sim_config configs/simulation_config.json --sel_coeffs
python mega_analysis.py Ube4b plots/ --sim_config configs/simulation_config.json --fitness --sel_coeffs

# Embedding-only analyses (cross_replicate_consistency, shuffled_frequencies)
python mega_analysis.py Ube4b plots/ --embedding_config configs/inference_config.json --cross_replicate_consistency

# Run all analyses (both configs required)
python mega_analysis.py Ube4b plots/ --sim_config configs/simulation_config.json --embedding_config configs/inference_config.json --all
"""

import os
import sys
import json
import pickle
import argparse
import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional
import pandas as pd

import numpy as np
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

pwd = "/net/dali/home/barton/dhw28/popDMS/esmDMS"
if not os.path.exists(pwd):
    pwd = "/Users/dylanwells/popDMS/esmDMS"
if pwd not in sys.path:
    sys.path.append(pwd)

from esmdmsfunctions import (
    get_simulation_results,
    generate_selection, gaussian_selection, zero_selection,
    load_final_df, load_inference_df,
    run_simulation as run_wright_fisher_simulation,
    run_inference_calcs_sims,
)

from analysis_helpers import (
    plot_true_vs_inferred_fitness,
    plot_true_vs_inferred_sel_coeffs,
    plot_cross_replicate_consistency,
    plot_shuffled_consistency,
    plot_fitness_trajectories,
    get_individual_fitness_values,
    get_esm_individual_fitness_values,
    get_enrichment_ratios,
    plot_baseline_vs_esm_comparison,
    plot_scatter_comparison,
)

from popDMS import mini_infer_independent_esm
from embedding_transforms import (
    EmbeddingTransform,
    IdentityTransform,
    make_transform,
    make_transforms,
    transform_embedding_df,
    unique_embeddings_from_df,
)

SEL_FUNC_MAP = {
    "gaussian": gaussian_selection,
    "generate": generate_selection,
    "zero":     zero_selection,
}

NCOLS = 6

TRANSFORM_CONFIG = {
    "identity": {},
    "sae": {
        "latent_dim": 64,
        "sparsity_weight": 1e-3,
        "learning_rate": 1e-3,
        "epochs": 200,
        "batch_size": 256,
        "activation_threshold": 1e-3,
        "random_state": 0,
    },
    "pca": {
        "variance_threshold": 0.95,
        "max_components": None,
        "random_state": 0,
    },
    "spca": {
        "n_components": 32,
        "alpha": 1.0,
        "ridge_alpha": 0.01,
        "random_state": 0,
        "max_iter": 500,
    },
    "ica": {
        "n_components": 32,
        "random_state": 0,
        "max_iter": 1000,
    },
}

EMBEDDING_TRANSFORMS = ("identity", "sae", "pca", "spca", "ica")


# ---------------------------------------------------------------------------
# Result objects + class-based pipeline
# ---------------------------------------------------------------------------

@dataclass
class TransformLayerResult:
    """popDMS outputs for one layer and one embedding transform."""

    layer: int
    transform_name: str
    s: np.ndarray
    s_joint: np.ndarray
    error_bars: np.ndarray
    s_joint_error_bars: np.ndarray
    icov: Any
    gamma: float
    feature_dim: int
    embedding_dim: int
    s_embedding: np.ndarray
    s_joint_embedding: np.ndarray
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    transform: Optional[EmbeddingTransform] = None

    def as_dict(self):
        return {
            "s": self.s,
            "s_joint": self.s_joint,
            "error_bars": self.error_bars,
            "s_joint_error_bars": self.s_joint_error_bars,
            "icov": self.icov,
            "gamma": self.gamma,
            "s_embedding": self.s_embedding,
            "s_joint_embedding": self.s_joint_embedding,
            "diagnostics": self.diagnostics,
            "feature_dim": self.feature_dim,
            "embedding_dim": self.embedding_dim,
            "transform": self.transform,
        }


@dataclass
class TransformRunResult:
    """Collection of per-layer results for one dataset/transform run."""

    dataset: str
    transform_name: str
    layers: Dict[int, TransformLayerResult]
    layer_fits: Optional[Dict[int, np.ndarray]] = None
    true_selection_coefficients: Optional[Dict[int, np.ndarray]] = None
    gamma_analysis: Optional[Dict[int, Any]] = None
    generation_counts: Optional[Dict[int, Any]] = None

    def as_modular_dict(self):
        return {layer: result.as_dict() for layer, result in self.layers.items()}

    def summary(self):
        rows = []
        for layer, result in self.layers.items():
            projected = result.s_joint_embedding
            projection_available = bool(np.all(np.isfinite(projected)))
            rows.append({
                "dataset": self.dataset,
                "transform": self.transform_name,
                "layer": layer,
                "feature_dim": result.feature_dim,
                "embedding_dim": result.embedding_dim,
                "gamma": result.gamma,
                "projection_available": projection_available,
                "s_joint_l2": float(np.linalg.norm(result.s_joint)),
                "s_joint_embedding_l2": (
                    float(np.linalg.norm(projected)) if projection_available else np.nan
                ),
            })
        return pd.DataFrame(rows)


def _fresh_transform(transform):
    if transform is None:
        return IdentityTransform()
    if isinstance(transform, str):
        return make_transform(transform, TRANSFORM_CONFIG.get(transform, {}))
    return copy.deepcopy(transform)


def _transform_name(transform):
    return getattr(transform, "name", transform.__class__.__name__.lower())


def _zscore(values):
    values = np.asarray(values, dtype=float)
    std = np.std(values)
    return values - np.mean(values) if std == 0 else (values - np.mean(values)) / std


def _fitness_from_scores(scores, fitness_fn):
    if fitness_fn == "exp":
        return np.exp(scores)
    if fitness_fn == "plus1":
        return np.maximum(1.0 + scores, 0.0)
    raise ValueError(f"Unknown fitness function: {fitness_fn}")


def _result_from_inference_data(layer, transform, data):
    s, s_joint, error_bars, s_joint_error_bars, icov, gamma_opt = (
        data[2], data[3], data[7], data[8], data[1], data[5]
    )
    n_original = getattr(transform, "n_features_in_", len(s_joint))
    projected_reps = []
    for s_rep in s:
        projected = transform.project_coefficients_to_embedding_space(s_rep)
        projected_reps.append(projected if projected is not None else np.full(n_original, np.nan))
    s_embedding = np.vstack(projected_reps)
    s_joint_embedding = transform.project_coefficients_to_embedding_space(s_joint)
    if s_joint_embedding is None:
        s_joint_embedding = np.full(n_original, np.nan)
    return TransformLayerResult(
        layer=layer,
        transform_name=_transform_name(transform),
        s=s,
        s_joint=s_joint,
        error_bars=error_bars,
        s_joint_error_bars=s_joint_error_bars,
        icov=icov,
        gamma=gamma_opt,
        feature_dim=int(len(s_joint)),
        embedding_dim=int(len(s_joint_embedding)),
        s_embedding=s_embedding,
        s_joint_embedding=s_joint_embedding,
        diagnostics=transform.diagnostics,
        transform=transform,
    )


class EmbeddingAnalysisDataset:
    """One dataset directory plus operations for transforms, popDMS, simulation, and plots."""

    def __init__(self, name, embedding_path, inference_cfg=None, simulation_cfg=None):
        self.name = name
        self.embedding_path = os.path.expanduser(str(embedding_path))
        self.inference_cfg = dict(inference_cfg or {})
        self.simulation_cfg = dict(simulation_cfg or {})

    def detect_layers(self, suffix=None):
        return _detect_layers(self.embedding_path, suffix=suffix)

    def inference_layers(self, layers=None):
        return list(layers if layers is not None else self.inference_cfg.get("layers") or self.detect_layers())

    def simulation_layers(self, layers=None):
        return list(layers if layers is not None else self.simulation_cfg.get("layers") or self.detect_layers("sim_df"))

    def load_inference_layer(self, layer, normalize=None, replicates=None):
        normalize = self.inference_cfg.get("normalize", "none") if normalize is None else normalize
        replicates = self.inference_cfg.get("replicates") if replicates is None else replicates
        return load_inference_df(layer, self.embedding_path, normalize=normalize, replicates=replicates)

    def load_simulation_layer(self, layer):
        return load_final_df(layer, self.embedding_path)

    def ensure_inference_files(self, layers=None):
        layers = self.inference_layers(layers)
        for layer in layers:
            has_compact = os.path.exists(os.path.join(self.embedding_path, f"layer{layer}_seq_to_emb.pkl"))
            has_legacy = os.path.exists(os.path.join(self.embedding_path, f"layer{layer}_inference_df.pkl"))
            if has_compact or has_legacy:
                continue
            directory_name = os.path.basename(self.embedding_path.rstrip("/"))
            emb_path = os.path.join(self.embedding_path, f"{directory_name}_embeddings.pkl")
            if not os.path.exists(emb_path):
                raise FileNotFoundError(
                    f"No layer data for layer {layer} in {self.embedding_path} and "
                    f"{directory_name}_embeddings.pkl not found."
                )
            print(f"Layer {layer} data missing — regenerating from {emb_path}", file=sys.stderr)
            embedding_df = pd.read_pickle(emb_path)
            embedding_df = embedding_df[embedding_df["Embedding"].notna()]
            emb_df_to_inference_dfs(embedding_df, self.embedding_path)
            break

    def ensure_simulation_files(self, layers=None):
        layers = self.simulation_layers(layers)
        for layer in layers:
            df_path = os.path.join(self.embedding_path, f"layer{layer}_sim_df.pkl")
            if os.path.exists(df_path):
                continue
            directory_name = os.path.basename(self.embedding_path.rstrip("/"))
            emb_path = os.path.join(self.embedding_path, f"{directory_name}_embeddings.pkl")
            if not os.path.exists(emb_path):
                raise FileNotFoundError(
                    f"Missing sim_df for layer {layer} and missing {directory_name}_embeddings.pkl at {emb_path}."
                )
            print(f"Layer {layer} sim data missing — regenerating from {emb_path}", file=sys.stderr)
            embedding_df = pd.read_pickle(emb_path)
            embedding_df = embedding_df[embedding_df["Embedding"].notna()]
            emb_df_to_sim_dfs(embedding_df, self.embedding_path)
            break

    def cache_path(self, kind, transform_name=None, cfg=None):
        cfg = cfg or self.inference_cfg
        normalize = cfg.get("normalize", "none")
        replicates = cfg.get("replicates")
        embedding_method = cfg.get("embedding_method")
        method_suffix = f"_{embedding_method}" if embedding_method else ""
        norm_suffix = f"_{normalize}" if normalize != "none" else ""
        rep_suffix = ("_reps" + "_".join(map(str, sorted(replicates)))) if replicates else ""
        transform_suffix = f"_{transform_name}" if transform_name else ""
        return os.path.join(
            self.embedding_path,
            f"{kind}{method_suffix}{norm_suffix}{rep_suffix}{transform_suffix}.pkl",
        )

    def run_inference(self, transform=None, layers=None, save_results=True, force_recompute=False, cache=True):
        """Fit one EmbeddingTransform per layer and run popDMS on transformed features."""
        base_transform = _fresh_transform(transform)
        transform_name = _transform_name(base_transform)
        layers = self.inference_layers(layers)
        cache_path = self.cache_path("inference_results", transform_name=transform_name)

        if cache and os.path.exists(cache_path) and not force_recompute:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            print(f"Inference results already exist at {cache_path}. Loading existing results.")
            return cached

        self.ensure_inference_files(layers)
        results = {}
        for layer in layers:
            layer_df = self.load_inference_layer(layer)
            fit_embeddings = unique_embeddings_from_df(layer_df)
            layer_transform = _fresh_transform(base_transform)
            print(f"[{self.name}] layer {layer}: fitting transform '{_transform_name(layer_transform)}'")
            layer_transform.fit(fit_embeddings)
            transformed_df = transform_embedding_df(layer_df, layer_transform)
            data = mini_infer_independent_esm(
                transformed_df,
                n_replicates=transformed_df["Replicate"].nunique(),
                gamma=None,
                corr_cutoff_pct=0.5,
                max_reads=1e3,
                output_dir=None,
                name=f"{self.name}_{_transform_name(layer_transform)}_inference",
                plot_gamma=False,
                verbose=True,
                calc_error_bars=False,
                variance_cutoff=0.0,
                infer_ignored_dims=True,
            )
            results[layer] = _result_from_inference_data(layer, layer_transform, data)
            del layer_df, transformed_df, data

        run_result = TransformRunResult(self.name, transform_name, results)
        if save_results:
            with open(cache_path, "wb") as f:
                pickle.dump(run_result, f, protocol=4)
            print(f"Inference results saved to {cache_path}")
        return run_result

    def run_inference_many(self, transforms=None, layers=None, save_results=True, force_recompute=False, cache=True):
        transforms = transforms or make_transforms(
            self.inference_cfg.get("embedding_transforms", EMBEDDING_TRANSFORMS),
            _merge_transform_config(self.inference_cfg),
        )
        return {
            _transform_name(_fresh_transform(transform)): self.run_inference(
                transform=transform,
                layers=layers,
                save_results=save_results,
                force_recompute=force_recompute,
                cache=cache,
            )
            for transform in transforms
        }

    def run_simulation(self, transform=None, layers=None, save_results=True, force_recompute=False, cache=True):
        """Run Wright-Fisher simulation and popDMS inference in a transform basis."""
        base_transform = _fresh_transform(transform)
        transform_name = _transform_name(base_transform)
        layers = self.simulation_layers(layers)
        cfg = self.simulation_cfg
        cache_path = self.cache_path("simulation_results", transform_name=transform_name, cfg=cfg)

        if cache and os.path.exists(cache_path) and not force_recompute:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            print(f"Simulation results already exist at {cache_path}. Loading existing results.")
            return cached

        self.ensure_simulation_files(layers)
        sel_func = SEL_FUNC_MAP[cfg.get("sel_func", "gaussian")]

        all_layer_fits = {}
        all_selection_coefficients = {}
        detailed_results = {}
        all_gamma_analysis = {}
        all_generation_counts = {}

        for layer in layers:
            print(f"[{self.name}] layer {layer}: running simulation with transform '{transform_name}'")
            df_selection = self.load_simulation_layer(layer)
            n_reps = sum(1 for c in df_selection.columns if c.endswith("_PreNums"))
            initial_counts = [df_selection[f"Rep{rep + 1}_PreNums"].values for rep in range(n_reps)]

            embeddings = np.vstack(df_selection["Embedding"].values)
            layer_transform = _fresh_transform(base_transform)
            layer_transform.fit(embeddings)
            transformed_embeddings = layer_transform.transform(embeddings)
            df_transformed = df_selection.copy()
            df_transformed["Embedding"] = [transformed_embeddings[i] for i in range(len(transformed_embeddings))]

            selection_coefficients = sel_func(transformed_embeddings.shape[1])
            generation_counts, layer_fits = run_wright_fisher_simulation(
                df_transformed,
                selection_coefficients,
                initial_counts,
                n_gens=cfg.get("n_gens", 30),
                save_every=cfg.get("save_every", 1),
                fitness=cfg.get("fitness_fn", "exp"),
                plateau_window=cfg.get("plateau_window", 0),
                plateau_rtol=cfg.get("plateau_rtol", 1e-3),
            )
            all_layer_fits[layer] = layer_fits
            all_selection_coefficients[layer] = selection_coefficients
            all_generation_counts[layer] = generation_counts

            data = run_inference_calcs_sims(
                df_transformed,
                generation_counts,
                calc_error_bars=cfg.get("calc_error_bars", False),
                variance_cutoff=cfg.get("variance_cutoff", 0.0),
                infer_ignored_dims=cfg.get("infer_ignored", True),
                method=cfg.get("method", "fullcov"),
            )
            detailed_results[layer] = _result_from_inference_data(layer, layer_transform, data)
            del df_selection, df_transformed, data

        run_result = TransformRunResult(
            dataset=self.name,
            transform_name=transform_name,
            layers=detailed_results,
            layer_fits=all_layer_fits,
            true_selection_coefficients=all_selection_coefficients,
            gamma_analysis=all_gamma_analysis,
            generation_counts=all_generation_counts,
        )
        if save_results:
            with open(cache_path, "wb") as f:
                pickle.dump(run_result, f, protocol=4)
            print(f"Simulation results saved to {cache_path}")
        return run_result

    def plot_standard(self, run_result, analyses, output_dir, cfg=None, paths=None):
        """Create standard plots for this dataset from a TransformRunResult."""
        cfg = cfg or self.inference_cfg
        paths = paths or {self.name: self.embedding_path}
        results_map = {self.name: run_result}
        os.makedirs(output_dir, exist_ok=True)
        for analysis in analyses:
            if analysis == "fitness" and run_result.transform_name != "identity":
                self._plot_transformed_fitness(run_result, cfg, output_dir)
                continue
            fn, _ = ANALYSES[analysis]
            if fn is None:
                continue
            fn(results_map, paths, cfg, output_dir)

    def _plot_transformed_fitness(self, run_result, cfg, output_dir):
        fitness_fn = cfg.get("fitness_fn", "exp") if cfg else "exp"
        layers = sorted(run_result.layers)
        n_layers = len(layers)
        n_cols = min(NCOLS, max(1, n_layers))
        n_rows = (n_layers + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows), squeeze=False)
        fig.suptitle(
            f"s_joint transformed-basis fitness — {self.name} / {run_result.transform_name}",
            fontsize=14,
            y=1.01,
        )
        for idx, layer in enumerate(layers):
            row, col = divmod(idx, n_cols)
            ax = axes[row, col]
            result = run_result.layers[layer]
            df_selection = self.load_simulation_layer(layer)
            embeddings = np.vstack(df_selection["Embedding"].values)
            transformed_embeddings = result.transform.transform(embeddings)
            inferred = _fitness_from_scores(transformed_embeddings @ result.s_joint, fitness_fn)
            true = np.asarray(run_result.layer_fits[layer])
            true_z = _zscore(true)
            inferred_z = _zscore(inferred)
            r = pearsonr(true_z, inferred_z)[0]
            ax.scatter(true_z, inferred_z, alpha=0.4, s=8, color="seagreen", rasterized=True)
            lo = min(true_z.min(), inferred_z.min()) - 0.3
            hi = max(true_z.max(), inferred_z.max()) + 0.3
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=0.8)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_title(f"Layer {layer} (r={r:.3f})", fontsize=9)
            ax.set_xlabel("True fitness (z)", fontsize=7)
            ax.set_ylabel("Inferred fitness (z)", fontsize=7)
            ax.tick_params(labelsize=7)
        for idx in range(n_layers, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row, col].set_visible(False)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"fitness_scatter_sjoint_{self.name}_{run_result.transform_name}.png"),
            dpi=90,
            bbox_inches="tight",
        )
        plt.close()

class MegaAnalysisPipeline:
    """Class-based driver for multi-dataset analyses and CLI orchestration."""

    def __init__(self, datasets, output_dir, inference_cfg=None, simulation_cfg=None):
        self.datasets = datasets
        self.output_dir = output_dir
        self.inference_cfg = dict(inference_cfg or {})
        self.simulation_cfg = dict(simulation_cfg or {})

    @classmethod
    def from_paths(cls, paths, output_dir, inference_cfg=None, simulation_cfg=None):
        datasets = {
            name: EmbeddingAnalysisDataset(name, path, inference_cfg, simulation_cfg)
            for name, path in paths.items()
        }
        return cls(datasets, output_dir, inference_cfg, simulation_cfg)

    def run_inference(self, transform=None, force_recompute=False):
        return {
            name: dataset.run_inference(transform=transform, force_recompute=force_recompute)
            for name, dataset in self.datasets.items()
        }

    def run_inference_many(self, transforms=None, force_recompute=False):
        return {
            name: dataset.run_inference_many(transforms=transforms, force_recompute=force_recompute)
            for name, dataset in self.datasets.items()
        }

    def run_simulation(self, transform=None, force_recompute=False):
        name, dataset = next(iter(self.datasets.items()))
        return {name: dataset.run_simulation(transform=transform, force_recompute=force_recompute)}

    def run_standard_analyses(self, selected, force_recompute=False, normalize="none", replicates=None):
        needs_sim = selected & set(ANALYSES_REQUIRING_SIM)
        needs_emb = selected & set(ANALYSES_REQUIRING_ONLY_EMB)
        modular_selected = "embedding_transform_comparison" in selected
        single_emb_selected = bool(needs_emb - {"embedding_transform_comparison"})

        all_results_sim = self.run_simulation(IdentityTransform(), force_recompute=force_recompute) if needs_sim else None
        all_results_emb = self.run_inference(IdentityTransform(), force_recompute=force_recompute) if single_emb_selected else None
        all_results_modular = self.run_inference_many(force_recompute=force_recompute) if modular_selected else None

        rep_label = ("reps" + "_".join(map(str, sorted(replicates)))) if replicates else None
        paths = {name: dataset.embedding_path for name, dataset in self.datasets.items()}

        ran_any = False
        for flag, (fn, description) in ANALYSES.items():
            if flag not in selected:
                continue
            if fn is None:
                print(f"Skipping '{flag}': not yet implemented.")
                continue
            print(f"Running: {description}")
            is_sim = flag in ANALYSES_REQUIRING_SIM
            if flag == "embedding_transform_comparison":
                all_results = all_results_modular
            elif is_sim:
                all_results = all_results_sim
            else:
                all_results = all_results_emb

            if is_sim:
                out_subdir = next(iter(self.datasets))
                cfg = self.simulation_cfg
            elif len(self.datasets) == 1:
                out_subdir = os.path.join(next(iter(self.datasets)), normalize, *([rep_label] if rep_label else []))
                cfg = self.inference_cfg
            else:
                out_subdir = os.path.join(normalize, *([rep_label] if rep_label else []))
                cfg = self.inference_cfg
            fn(all_results, paths, cfg, output_dir=os.path.join(self.output_dir, out_subdir))
            ran_any = True
        if not ran_any:
            print("Warning: no analyses ran (all selected analyses may be unimplemented).")


# ---------------------------------------------------------------------------
# Config + simulation
# ---------------------------------------------------------------------------

def load_config(config_path, dataset=None):
    with open(config_path) as f:
        cfg = json.load(f)
    if dataset is not None:
        values = {"dataset": dataset, "embedding_method": cfg.get("embedding_method", "mean_pool")}
        cfg = {
            k: os.path.expanduser(v.format(**values)) if isinstance(v, str) else v
            for k, v in cfg.items()
        }
    return cfg


def _detect_layers(embedding_path, suffix=None):
    """Return sorted layer indices by scanning for layer{i}_{suffix}.pkl files.

    When suffix is None (inference case), tries 'seq_to_emb' first (new compact
    format) then 'inference_df' (legacy format) so both layouts are handled.
    """
    import re
    try:
        files = os.listdir(embedding_path)
    except OSError:
        return list(range(31))
    for s in ([suffix] if suffix else ["seq_to_emb", "inference_df"]):
        layers = sorted(
            int(m.group(1))
            for f in files
            if (m := re.match(rf"layer(\d+)_{s}\.pkl", f))
        )
        if layers:
            return layers
    return list(range(31))


def emb_df_to_sim_dfs(emb_df, out_path):
    # Format is Embeddings , PreNum_1, PreNum_2, ...
    # Out format will save a dataframe for each layer
    reps = sorted(emb_df["Replicate"].unique())
    pre_df = emb_df[emb_df["Generation"] == 0]

    key_cols = ["ProteinSequence"] + [
        col for col in ["MutationSiteIndex", "MutationSite"]
        if col in emb_df.columns
    ]

    # One row per unique embedding unit (sequence, or sequence/site for residue embeddings)
    result = (
        pre_df.drop_duplicates(subset=key_cols)
              .set_index(key_cols)[["Embedding"]]
    )

    for rep in reps:
        rep_freqs = pre_df[pre_df["Replicate"] == rep].set_index(key_cols)["Frequency"]
        result[f"Rep{rep}_PreNums"] = rep_freqs

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    num_layers = len(result["Embedding"].iloc[0])
    for layer in range(num_layers):
        layer_df = result[["Embedding", *[col for col in result.columns if col.startswith("Rep")]]].copy()
        layer_df["Embedding"] = layer_df["Embedding"].apply(lambda x: x[layer])
        # save the pickle file
        out_path_layer = os.path.join(out_path, f"layer{layer}_sim_df.pkl")
        layer_df.drop(columns=["ProteinSequence"], inplace=True, errors="ignore")
        layer_df.to_pickle(out_path_layer)
        print(f"Saved layer {layer} dataframe to {out_path_layer}")
    print("All layers saved successfully.")


def emb_df_to_inference_dfs(emb_df, out_path):
    """Save compact per-layer inference files.

    Instead of duplicating each sequence's embedding across every (replicate,
    generation) row, we store:
      - layer{i}_seq_to_emb.pkl : (n_unique_seqs, emb_dim) numpy array indexed by seq_id
      - layer{i}_inference_df.pkl: metadata df with (seq_id, Replicate, Generation, Frequency)
      - seq_id_map.pkl           : list of ProteinSequence strings in seq_id order

    load_inference_df() reconstructs the full df with an Embedding column on load.
    This cuts per-layer file size from O(n_rows x emb_dim) to O(n_unique x emb_dim).
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    num_layers = len(emb_df["Embedding"].iloc[0])

    key_cols = ["ProteinSequence"] + [
        col for col in ["MutationSiteIndex", "MutationSite"]
        if col in emb_df.columns
    ]

    # One row per unique embedding unit for compact per-layer arrays.
    unique_embs = emb_df.drop_duplicates(key_cols).reset_index(drop=True)
    def _stable_key_value(value):
        if isinstance(value, list):
            return tuple(value)
        if isinstance(value, np.ndarray):
            return tuple(value.tolist())
        if not isinstance(value, (tuple, list, dict)):
            try:
                if pd.isna(value):
                    return None
            except (TypeError, ValueError):
                pass
        return value

    unique_keys = unique_embs[key_cols].to_dict("records")
    key_tuples = [tuple(_stable_key_value(row[col]) for col in key_cols) for row in unique_keys]
    key_to_id = {key: i for i, key in enumerate(key_tuples)}

    # Compact metadata df: integer seq_id replaces both ProteinSequence and Embedding
    meta_cols = [*key_cols, "Replicate", "Generation", "Frequency"]
    for extra_col in ["MutationSites", "EmbeddingMethod", "PoolMutations"]:
        if extra_col in emb_df.columns and extra_col not in meta_cols:
            meta_cols.insert(len(key_cols), extra_col)
    meta_df = emb_df[meta_cols].copy()
    meta_keys = [tuple(_stable_key_value(row[col]) for col in key_cols) for _, row in meta_df.iterrows()]
    meta_df["seq_id"] = [key_to_id[key] for key in meta_keys]
    meta_df = meta_df.drop(columns=["ProteinSequence"]).reset_index(drop=True)

    with open(os.path.join(out_path, "seq_id_map.pkl"), "wb") as f:
        pickle.dump(unique_keys, f)
    print(f"Saved seq_id_map ({len(unique_keys):,} unique embedding units)")

    # Save metadata once — it is identical across all layers (seq_id, Replicate, Generation, Frequency).
    # load_inference_df checks for this file first; per-layer copies are kept for backward compatibility
    # with any existing data but are no longer written for new saves.
    meta_df.to_pickle(os.path.join(out_path, "inference_metadata.pkl"))
    print(f"Saved inference_metadata ({len(meta_df):,} rows, shared across all layers)")

    for layer in range(num_layers):
        layer_embs = np.array([emb[layer] for emb in unique_embs["Embedding"]])  # (n_unique, emb_dim)

        seq_to_emb_path = os.path.join(out_path, f"layer{layer}_seq_to_emb.pkl")
        with open(seq_to_emb_path, "wb") as f:
            pickle.dump(layer_embs, f)

        print(f"Saved layer {layer}: {layer_embs.nbytes / 1e6:.0f} MB (seq_to_emb)")
    print("All layers saved successfully.")


def _merge_transform_config(inference_cfg):
    cfg = {name: dict(params) for name, params in TRANSFORM_CONFIG.items()}
    for name, params in inference_cfg.get("transform_config", {}).items():
        cfg.setdefault(name, {}).update(params)
    return cfg


def run_modular_inference(embedding_path, inference_cfg, save_results=True, force_recompute=False, cache=True):
    """Run popDMS once per registered embedding transform.

    Returns
    -------
    dict
        {transform_name: TransformRunResult} keyed by transform name.
    """
    dataset = EmbeddingAnalysisDataset(
        name=os.path.basename(embedding_path.rstrip("/")) or "dataset",
        embedding_path=embedding_path,
        inference_cfg=inference_cfg,
    )
    return dataset.run_inference_many(
        save_results=save_results,
        force_recompute=force_recompute,
        cache=cache,
    )


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def plot_cross_replicate_consistency_analysis(all_results, paths, cfg, output_dir):
    embedding_path = next(iter(paths.values()))
    layers = cfg.get("layers") or _detect_layers(embedding_path)
    plot_cross_replicate_consistency(all_results, layers, output_dir=output_dir,
                                     normalize=cfg.get("normalize", "none"),
                                     every_n=cfg.get("every_n", 1))



def plot_shuffled_frequencies_analysis(all_results, paths, cfg, output_dir):
    """Cross-replicate consistency after shuffling frequencies within each replicate.

    For each layer, independently shuffles the pre- and post-selection frequency
    values within each replicate (breaking the embedding→count association), then
    re-runs inference and plots cross-replicate consistency. A high r on real data
    but low r here confirms the signal is not an artefact of count structure.
    """
    normalize  = cfg.get("normalize", "none")
    replicates = cfg.get("replicates")
    rng = np.random.default_rng()
    shuffled_all_results = {}

    layers = None

    for name, embedding_path in paths.items():
        layers = cfg.get("layers") or _detect_layers(embedding_path)
        shuffled_layers = {}

        for layer in layers:
            layer_df = load_inference_df(layer, embedding_path, normalize=normalize, replicates=replicates)

            # Shuffle frequencies independently within each (Replicate, Generation) group
            for _, group_idx in layer_df.groupby(["Replicate", "Generation"]).groups.items():
                freqs = layer_df.loc[group_idx, "Frequency"].values.copy()
                rng.shuffle(freqs)
                layer_df.loc[group_idx, "Frequency"] = freqs

            n_reps = layer_df["Replicate"].nunique()
            data = mini_infer_independent_esm(
                layer_df, n_replicates=n_reps, gamma=None, corr_cutoff_pct=0.5,
                max_reads=1e3, output_dir=None, name="esm_inference_shuffled",
                plot_gamma=False, verbose=False, calc_error_bars=False,
                variance_cutoff=0.0, infer_ignored_dims=True,
            )
            shuffled_layers[layer] = _result_from_inference_data(layer, IdentityTransform(), data)
            del layer_df, data

        shuffled_all_results[name] = TransformRunResult(name, "identity_shuffled", shuffled_layers)

    shuffled_output_dir = os.path.join(output_dir, "shuffled")
    plot_cross_replicate_consistency(shuffled_all_results, layers, output_dir=shuffled_output_dir,
                                     normalize=normalize, every_n=cfg.get("every_n", 1))


def plot_shuffled_consistency_analysis(all_results, paths, cfg, output_dir):
    """For each dataset, run shuffled-frequency inference and collect per-layer
    mean cross-replicate correlations. Then plot whether the same layers show
    consistently high correlations across datasets.

    A high cross-dataset profile correlation means the spurious signal is
    structural (e.g. count depth or embedding geometry); low correlation means
    it is noise that varies independently per experiment.
    """
    normalize  = cfg.get("normalize", "none")
    replicates = cfg.get("replicates")
    rng = np.random.default_rng()

    # {dataset_name: {layer: (mean_r, std_r)}}
    layer_stats = {}

    for name, embedding_path in paths.items():
        layers = cfg.get("layers") or _detect_layers(embedding_path)
        stats_this = {}

        for layer in layers:
            layer_df = load_inference_df(layer, embedding_path, normalize=normalize, replicates=replicates)

            for _, group_idx in layer_df.groupby(["Replicate", "Generation"]).groups.items():
                freqs = layer_df.loc[group_idx, "Frequency"].values.copy()
                rng.shuffle(freqs)
                layer_df.loc[group_idx, "Frequency"] = freqs

            n_reps = layer_df["Replicate"].nunique()
            data = mini_infer_independent_esm(
                layer_df, n_replicates=n_reps, gamma=None, corr_cutoff_pct=0.5,
                max_reads=1e3, output_dir=None, name="esm_inference_shuffled",
                plot_gamma=False, verbose=False, calc_error_bars=False,
                variance_cutoff=0.0, infer_ignored_dims=True,
            )
            s_reps = data[2]
            del layer_df, data

            off_diag = [
                pearsonr(s_reps[i], s_reps[j])[0]
                for i in range(len(s_reps))
                for j in range(i + 1, len(s_reps))
            ]
            stats_this[layer] = (float(np.mean(off_diag)), float(np.std(off_diag)))

        layer_stats[name] = stats_this
        print(f"[{name}] shuffled consistency computed for {len(stats_this)} layers")

    all_layers = sorted(set().union(*(s.keys() for s in layer_stats.values())))
    plot_shuffled_consistency(layer_stats, all_layers, output_dir=output_dir, normalize=normalize)


def popDMS_esmDMS_comparison_analysis(all_results, paths, cfg, output_dir):
    """Compare the inferred fitness of every individual within
    the embedding-based inference to the fitness inferred by popDMS on the same data.
    """
    pop_inference_path    = cfg.get("pop_inference_path", None)
    reference_sequence_file = cfg.get("reference_sequence_file", None)
    haplotype_counts_file = cfg.get("haplotype_counts_file", None)
    fitness_fn            = cfg.get("fitness_fn", "plus1")
    comment_char          = cfg.get("comment_char", None)

    if not pop_inference_path or not os.path.exists(pop_inference_path):
        print(f"popDMS inference file not found at {pop_inference_path}. "
              "Run popDMS inference first (e.g. via the data_analysis notebook).")
        return

    with open(reference_sequence_file) as f:
        reference_sequence = f.read().strip()

    normalize = cfg.get("normalize", "none")

    # Part 1: load per-haplotype popDMS fitness values (same model as ESM-DMS)
    popdms_fits = get_individual_fitness_values(
        pop_inference_path, haplotype_counts_file, reference_sequence,
        fitness_fn=fitness_fn, comment_char=comment_char,
    )

    os.makedirs(output_dir, exist_ok=True)

    for path_name, run_result in all_results.items():
        embedding_path = paths[path_name]
        layers = sorted(run_result.layers.keys())

        # Part 2: for each layer, compute ESM-DMS inferred fitness per sequence.
        # normalize must match what was used during inference so embeddings are
        # in the same space as s_joint.
        esm_fits_by_layer = {}
        for layer in layers:
            s_joint = run_result.layers[layer].s_joint
            esm_fits_by_layer[layer] = get_esm_individual_fitness_values(
                embedding_path, layer, s_joint, fitness_fn=fitness_fn,
                normalize=normalize,
            )

        # Part 3: plot comparisons
        plot_baseline_vs_esm_comparison(
            popdms_fits, esm_fits_by_layer, layers, output_dir, path_name,
            baseline_label="popDMS Fitness",
        )
        enrichment_ratios = get_enrichment_ratios(embedding_path)
        plot_baseline_vs_esm_comparison(
            enrichment_ratios, esm_fits_by_layer, layers, output_dir, path_name,
            baseline_label="Enrichment Ratio",
        )


def popDMS_enrichment_comparison_analysis(all_results, paths, cfg, output_dir):
    """Scatter plot comparing popDMS inferred fitness directly to log enrichment ratio."""
    pop_inference_path    = cfg.get("pop_inference_path", None)
    reference_sequence_file = cfg.get("reference_sequence_file", None)
    haplotype_counts_file = cfg.get("haplotype_counts_file", None)
    fitness_fn            = cfg.get("fitness_fn", "plus1")
    comment_char          = cfg.get("comment_char", None)

    if not pop_inference_path or not os.path.exists(pop_inference_path):
        print(f"popDMS inference file not found at {pop_inference_path}. "
              "Run popDMS inference first (e.g. via the data_analysis notebook).")
        return

    with open(reference_sequence_file) as f:
        reference_sequence = f.read().strip()

    popdms_fits = get_individual_fitness_values(
        pop_inference_path, haplotype_counts_file, reference_sequence,
        fitness_fn=fitness_fn, comment_char=comment_char,
    )

    os.makedirs(output_dir, exist_ok=True)

    for path_name, _ in all_results.items():
        embedding_path = paths[path_name]
        enrichment_ratios = get_enrichment_ratios(embedding_path)
        plot_scatter_comparison(
            popdms_fits, enrichment_ratios, output_dir, path_name,
            x_label="popDMS Fitness", y_label="Enrichment Ratio",
        )


def embedding_transform_comparison_analysis(all_results, paths, cfg, output_dir):
    """Write unified transform comparison outputs for modular popDMS runs.

    ``all_results`` is ``dict[dataset_name -> dict[transform_name -> TransformRunResult]]``.
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []

    for dataset, transform_results in all_results.items():
        for transform_name, run_result in transform_results.items():
            for layer, layer_result in run_result.layers.items():
                projected = layer_result.s_joint_embedding
                projection_available = bool(np.all(np.isfinite(projected)))
                summary_rows.append({
                    "dataset": dataset,
                    "transform": transform_name,
                    "layer": layer,
                    "feature_dim": layer_result.feature_dim,
                    "embedding_dim": layer_result.embedding_dim,
                    "gamma": layer_result.gamma,
                    "projection_available": projection_available,
                    "s_joint_l2": float(np.linalg.norm(layer_result.s_joint)),
                    "s_joint_embedding_l2": (
                        float(np.linalg.norm(projected)) if projection_available else np.nan
                    ),
                })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(output_dir, "embedding_transform_summary.csv"), index=False)
    with open(os.path.join(output_dir, "embedding_transform_results.pkl"), "wb") as f:
        pickle.dump(all_results, f, protocol=4)
    print(f"Wrote transform comparison summary to {output_dir}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ANALYSES = {
    # Simulation-based analyses
    "fitness": (
        plot_true_vs_inferred_fitness,
        "true vs inferred fitness",
    ),
    "sel_coeffs": (
        plot_true_vs_inferred_sel_coeffs,
        "true vs inferred selection coefficients",
    ),
    "gamma": (
        None,  # placeholder for future gamma analysis function
        "gamma analysis (not implemented yet)",
    ),
    "sparsify": (
        None,  # placeholder for future sparsification analysis function
        "sparsification analysis (not implemented yet)",
    ),
    "eigenvalue_distribution": (
        None,  # placeholder for future eigenvalue distribution analysis function
        "eigenvalue distribution analysis (not implemented yet)",
    ),
    # Embedding-only analyses
    "cross_replicate_consistency": (
        plot_cross_replicate_consistency_analysis,
        "Plot the cross-replicate consistency for inferred selection coefficients across layers",
    ),
    "shuffled_frequencies": (
        plot_shuffled_frequencies_analysis,
        "Run cross-replicate consistency analysis on data with shuffled post-selection frequencies",
    ),
    "shuffled_consistency": (
        plot_shuffled_consistency_analysis,
        "Compare per-layer shuffled correlations across multiple datasets to identify structural artifacts",
    ),
    "popDMS_comparison": (
        popDMS_esmDMS_comparison_analysis,
        "Compare per-individual ESM-DMS inferred fitness to popDMS fitness",
    ),
    "popDMS_enrichment_comparison": (
        popDMS_enrichment_comparison_analysis,
        "Scatter plot comparing popDMS inferred fitness directly to log enrichment ratio",
    ),
    "embedding_transform_comparison": (
        embedding_transform_comparison_analysis,
        "Run registered embedding transforms through popDMS and compare coefficient projections",
    ),
}

# Now categorize the analyses by which need simulation results vs just embeddings

ANALYSES_REQUIRING_SIM = ("fitness", "sel_coeffs")
ANALYSES_REQUIRING_ONLY_EMB = tuple(k for k in ANALYSES if k not in ANALYSES_REQUIRING_SIM)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run simulation and modular analyses on an embedding dataframe."
    )
    parser.add_argument("dataset", nargs="+",
                        help="Dataset name(s) (e.g. Ube4b); substituted for {dataset} in all config paths. "
                             "Pass multiple names for multi-dataset analyses like --shuffled_consistency.")
    parser.add_argument("--sim_config", help="Path to JSON config file (simulation parameters, optional)")
    parser.add_argument("--embedding_config", help="Path to JSON config file (embedding parameters, optional)")
    parser.add_argument("output_dir", help="Directory to write output plots")

    analysis_group = parser.add_argument_group("analyses (at least one required)")
    analysis_group.add_argument("--all", dest="run_all", action="store_true",
                                help="Run all analyses")
    for flag, (_, description) in ANALYSES.items():
        analysis_group.add_argument(f"--{flag}", action="store_true",
                                    help=f"Plot {description}")

    parser.add_argument("--force_recompute", action="store_true",
                        help="Ignore cached inference_results.pkl and recompute from scratch")
    parser.add_argument("--normalize", choices=["none", "by_layer", "by_layer_dim"], default="none",
                        help="Normalize embeddings before inference: none (default), by_layer (global z-score per layer), by_layer_dim (per-dimension z-score per layer)")
    parser.add_argument("--replicates", nargs="+", type=int, default=None, metavar="REP",
                        help="Restrict inference to a subset of replicate IDs, e.g. --replicates 1 2 3 6 7 8")
    parser.add_argument("--every_n", type=int, default=None, metavar="N",
                        help="Only produce per-layer plots every N layers (default: 1, i.e. every layer)")
    parser.add_argument("--embedding_method", choices=["mean_pool", "cls", "mutation_site"], default=None,
                        help="Embedding extraction method label used by configs with {embedding_method} paths")

    args = parser.parse_args()

    datasets = args.dataset  # list of one or more dataset names

    selected = {flag for flag in ANALYSES if args.run_all or getattr(args, flag, False)}
    if not selected:
        parser.error("No analyses selected. Pass --all or one or more analysis flags.")

    needs_sim = selected & set(ANALYSES_REQUIRING_SIM)
    needs_emb = selected & set(ANALYSES_REQUIRING_ONLY_EMB)

    if needs_sim and not args.sim_config:
        parser.error(f"--sim_config is required for: {', '.join(needs_sim)}")
    if needs_emb and not args.embedding_config:
        parser.error(f"--embedding_config is required for: {', '.join(needs_emb)}")

    # Simulation analyses only support a single dataset; use the first one.
    primary_dataset = datasets[0]
    sim_cfg = load_config(args.sim_config, primary_dataset) if args.sim_config else {}

    # Base emb config (normalize/layers/etc.) from first dataset; path overridden per dataset below.
    emb_cfg = load_config(args.embedding_config, primary_dataset) if args.embedding_config else {}
    emb_cfg["normalize"]  = args.normalize
    emb_cfg["replicates"] = args.replicates
    if args.embedding_method is not None:
        emb_cfg["embedding_method"] = args.embedding_method
    if args.every_n is not None:
        emb_cfg["every_n"] = args.every_n

    # Build paths dict: resolve embedding_path for every dataset.
    paths = {}
    for ds in datasets:
        cfg_ds = load_config(args.embedding_config, ds) if args.embedding_config else {}
        if args.embedding_method is not None:
            cfg_ds["embedding_method"] = args.embedding_method
            if isinstance(cfg_ds.get("embedding_path"), str) and "{embedding_method}" in cfg_ds["embedding_path"]:
                cfg_ds["embedding_path"] = os.path.expanduser(
                    cfg_ds["embedding_path"].format(dataset=ds, embedding_method=args.embedding_method)
                )
        emb_path = cfg_ds.get("embedding_path") or sim_cfg.get("embedding_path")
        if emb_path:
            paths[ds] = emb_path
    if not paths:
        parser.error("embedding_path not found in any config file.")

    pipeline = MegaAnalysisPipeline.from_paths(
        paths=paths,
        output_dir=args.output_dir,
        inference_cfg=emb_cfg,
        simulation_cfg=sim_cfg,
    )
    pipeline.run_standard_analyses(
        selected=selected,
        force_recompute=args.force_recompute,
        normalize=args.normalize,
        replicates=args.replicates,
    )


if __name__ == "__main__":
    main()
