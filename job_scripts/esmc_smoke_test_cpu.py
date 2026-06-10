"""CPU-only smoke test for ESMC on the big_memory partition.

Loads the ESMC model named in ESMC_MODEL (default biohub/ESMC-6B), embeds a
short test protein on CPU, and prints device / dtype / shape.

Submit:
    sbatch job_scripts/submit_esmc_smoke_test_cpu.sh
Interactive (300M for a quick check):
    srun -p big_memory --mem=16G --pty python3 job_scripts/esmc_smoke_test_cpu.py
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from esmDMS import esmDMS  # noqa: E402

MODEL = os.environ.get("ESMC_MODEL", "biohub/ESMC-6B")
GFP = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSY"
    "GVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKE"
    "DGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDN"
    "HYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)


def main() -> int:
    print(f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
    print("running on CPU")
    print(f"loading {MODEL} (dtype env={os.environ.get('ESMDMS_TORCH_DTYPE', '<default>')})")

    # Ensure no GPU is used even if one happens to be visible.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    tokenizer, model = esmDMS._load_embedding_model(MODEL)
    param = next(model.parameters())
    print(f"model device={param.device}, dtype={param.dtype}")

    emb = esmDMS._embed_sequence(GFP, tokenizer, model)
    print(f"residue embeddings shape: {emb.shape}  (num_residues, num_layers, embedding_dim)")
    print(f"first-residue, first-layer norm: {float((emb[0, 0] ** 2).sum() ** 0.5):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
