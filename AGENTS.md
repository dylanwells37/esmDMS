# Codex Instructions

## Change Control

Never change code, notebooks, scripts, configs, job files, or other repository files unless the user explicitly asks for a change. For conceptual, debugging, review, or explanatory questions, answer without editing files. If a code change seems useful but was not explicitly requested, ask for confirmation first.

## Scientific and Mathematical Change Control

Never change a scientific assumption, physical or statistical model, equation, objective function, parameterization, approximation, optimizer, solver, or numerical algorithm without first explaining the proposed change and obtaining the user's explicit approval. This requirement applies even when the proposed implementation is believed to be mathematically equivalent or purely numerical. Before requesting approval, state exactly what remains invariant, what changes computationally, why the change is being considered, its numerical risks, and how equivalence or correctness would be validated. When diagnosing a failure, do not edit the affected implementation until that approval is given.
