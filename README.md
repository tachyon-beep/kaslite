# kaslite

This repo demonstrates a morphogenetic architecture with "soft-landing" seeds. Each sentinel seed now awakens gradually: it shadow-trains as an auto-encoder, blends its output into the trunk using a ramped alpha parameter, then becomes fully active. Command line flags `--blend_steps`, `--shadow_lr` and `--progress_thresh` control the blend duration, shadow learning rate and training-progress threshold respectively when running `scripts/run_spirals.py`.

## Changelog
- Hardened soft-landing controller:
  - buffer sampling guard
  - blocked gradient leakage
  - synced status with state
  - CLI-tunable drift warnings
