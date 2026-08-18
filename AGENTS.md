# AGENTS.md

## Project

This project experimentally renders DOOM frames through the internal
activation space of a frozen transformer.

Read `docs/PROJECT_BRIEF.md` before making architectural decisions.

## Working style

This is an experimental research/software project.

For substantial features:

1. Inspect the repository first.
2. Read `docs/PROJECT_BRIEF.md`.
3. Create or update `docs/EXECPLAN.md`.
4. Plan before implementing.
5. Work milestone-by-milestone.
6. Run the relevant tests after each milestone.
7. Prefer small working prototypes over prematurely complex abstractions.
8. Record important experimental findings and architectural decisions in
   `docs/EXECPLAN.md`.

Do not silently change the scientific meaning of the experiment.

## Core scientific constraint

The transformer should remain frozen unless explicitly stated otherwise.

The project is about transformer ACTIVATIONS, not changing the model weights.

A DOOM frame should genuinely be represented by values derived from an
internal transformer activation tensor. Do not simply overlay a DOOM image
on top of an unrelated heatmap.

## Engineering

Use Python.

Prefer:
- PyTorch
- Hugging Face Transformers
- NumPy
- matplotlib initially
- ViZDoom when real game frames are introduced

Keep model-specific code isolated behind a small interface so we can later
try different transformer models.

CUDA should be used when available, but the code should fail gracefully or
fall back to CPU where practical.

Keep experiments reproducible with explicit random seeds and configuration.

## Verification

For each implementation milestone:

- demonstrate that the code actually runs;
- report tensor shapes;
- validate gradients where gradients are expected;
- validate that transformer parameters remain frozen;
- save representative visualization artifacts;
- add lightweight tests for non-experimental utilities.

Do not claim success solely because the program executes.