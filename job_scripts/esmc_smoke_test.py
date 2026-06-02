"""Self-contained GPU + ESMC smoke test.

Loads the ESMC model named in ESMC_MODEL (default biohub/ESMC-300M for a
fast test; switch to biohub/ESMC-6B once that passes), embeds a short
test protein, and prints device / dtype / shape so you can confirm the
GPU is actually being used.

Run from a GPU node:
    sbatch job_scripts/submit_esmc_smoke_test.sh
or interactively:
    srun -p dept_gpu --gres=gpu:1 --pty python3 job_scripts/esmc_smoke_test.py
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from esmDMS import esmDMS  # noqa: E402

MODEL = os.environ.get("ESMC_MODEL", "biohub/ESMC-300M")
GFP = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSY"
    "GVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKE"
    "DGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDN"
    "HYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)


def main() -> int:
    print(f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}  ({torch.cuda.device_count()} visible)")
    print(f"loading {MODEL} (dtype env={os.environ.get('ESMDMS_TORCH_DTYPE', '<default>')})")

    tokenizer, model = esmDMS._load_embedding_model(MODEL)
    param = next(model.parameters())
    print(f"model device={param.device}, dtype={param.dtype}")

    emb = esmDMS._embed_sequence(GFP, tokenizer, model)
    print(f"layer embeddings shape: {emb.shape}  (num_layers, embedding_dim)")
    print(f"first-layer norm: {float((emb[0] ** 2).sum() ** 0.5):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
