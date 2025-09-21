# Simple SD 3.5 Medium Test for Koyeb

## GitHub Issue #1042: Add model: Stable Diffusion 3.5 medium (512x512)

Simple test to measure performance:
- Target: 0.3 FPS on batch 1
- Baseline: 0.06 FPS on batch 1

## Koyeb Settings:
- Buildpack: Python
- Run Command: `python test_sd35.py`
- Instance: Tenstorrent Wormhole/Blackhole
- Memory: 16 GiB
