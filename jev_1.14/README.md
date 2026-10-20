# Jev 1.14 + PEBBLE

This is the skill-conditioned PEBBLE training from `jev-pebble-skill`, with the judge swapped to `typesafe/jev-1.14`.

The skill run sent a textual observation and action trajectory to Jev 1.13. This run does not. Each preference query renders the two clips and sends a few camera frames in time order (earliest first). Jev 1.14 answers the same A / B / tie choice, and that label trains the reward model.

A tie is kept only when its probability is greater than 60% (`--jev-tie-threshold`, default 0.60). Otherwise the label is whichever of A or B has the higher probability, and that side is the reward target.

The OpenRouter API key is not stored here. Export it before training:

```bash
cd /data/nazari/rlvlmf/training-cnn/jev_1.14
pip install -e .
export OPENROUTER_API_KEY=your_key
python -m jev114.train --output runs_jev114
```

Use the same environment that ran `jev-pebble-skill` (MetaWorld, dm_control, torch). On a headless machine set `MUJOCO_GL=egl`; the trainer does that if it is unset.

Useful knobs:

```bash
python -m jev114.train --output runs_jev114 \
  --env sweep --frames-per-clip 8 --camera corner --render-width 480 --render-height 480
```

`--frames-per-clip` is how many frames of each clip Jev sees. They are spaced across the segment so the short sequence covers the motion. Frame size must stay inside Jev's image limits: 256 to 1568 pixels on each side, and at least 280×280 for a square image.
