# Jev 1.14 sweep-into preference test

Judges sweep-into A/B pairs with `typesafe/jev-1.14`.
Each side sends 8 of its 10 saved frames. The task sentence is:

> Sweep the object into the target area with the robot end effector.

Checkpoints 40 feedback labels apart (35 pairs, 70 trajectories):

```bash
cd /data/nazari/rlvlmf/training-cnn/jev1.14test
export OPENROUTER_API_KEY=your_key
python3 run_sweep_preferences.py --feedback-stride 40
```

Results append to `results_stride40.jsonl`. A second run skips pairs already in that file.
