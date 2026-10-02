# DATA-11 r4train: training step throughput with synthetic batches

- lerobot_sha: e0d50211ef236143ae867228662b7dfaba554f02
- python: 3.12.3
- host: j-hadriencornier-6abc66bb4c46ef1987033e62-s3pl1i3q-ec066-9sfdm
- warmup: 10
- steps: 50
- device: cuda
- gpu: NVIDIA L4
- torch: 2.11.0+cu128 cuda 12.8 cudnn 91900

update = forward + backward + grad clip + optimizer + scheduler (`lerobot_train.update_policy`).
prep = uint8->float on CPU + policy preprocessor (tokenizer, device copy, normalization).
samples/s = batch / median update. Camera frames/s = samples/s x cameras.

| policy | setup | cams | batch | status | update median ms | update p90 ms | prep median ms | samples/s | cam frames/s | samples/s incl prep | peak alloc GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| act | abc | 3 | 8 | ok | 67.5 | 68.4 | 5.4 | 118.5 | 355.5 | 109.5 | 1.85 |
| act | droid | 3 | 8 | ok | 72.7 | 73.6 | 6.1 | 110.0 | 330.1 | 101.7 | 1.89 |
| act | molmo | 3 | 8 | ok | 246.4 | 247.7 | 22.7 | 32.5 | 97.4 | 29.7 | 4.36 |
| act | libero | 2 | 8 | ok | 60.6 | 61.4 | 4.9 | 132.0 | 264.0 | 121.7 | 1.81 |
| act | hqf | 3 | 8 | ok | 1010.8 | 1013.6 | 94.2 | 7.9 | 23.7 | 7.3 | 18.99 |
| smolvla | abc | 3 | 8 | ok | 316.1 | 319.5 | 3.5 | 25.3 | 75.9 | 25.0 | 3.56 |
| smolvla | droid | 3 | 8 | ok | 315.4 | 318.2 | 3.8 | 25.4 | 76.1 | 25.1 | 3.57 |
| smolvla | molmo | 3 | 8 | ok | 309.6 | 311.3 | 23.4 | 25.8 | 77.5 | 24.0 | 3.61 |
| smolvla | libero | 2 | 8 | ok | 218.6 | 221.6 | 3.0 | 36.6 | 73.2 | 36.1 | 2.91 |
| smolvla | hqf | 3 | 8 | ok | 310.3 | 311.3 | 90.6 | 25.8 | 77.3 | 20.0 | 3.74 |
| smolvla | abc | 3 | 64 | ok | 2872.9 | 2881.4 | 65.9 | 22.3 | 66.8 | 21.8 | 19.32 |
| smolvla | droid | 3 | 64 | ok | 2871.1 | 2886.7 | 74.3 | 22.3 | 66.9 | 21.8 | 19.33 |
| smolvla | molmo | 3 | 64 | ok | 2850.8 | 2880.1 | 263.4 | 22.4 | 67.3 | 20.6 | 19.69 |
| smolvla | libero | 2 | 64 | skipped_budget | - | - | - | - | - | - | - |
| smolvla | hqf | 3 | 64 | skipped_budget | - | - | - | - | - | - | - |

## Defaults found (per policy)

- act: `{"train_batch_size_default": 8, "mixed_precision": "no", "gradient_accumulation_steps": 1, "policy_use_amp_field": false, "use_policy_training_preset": true, "optimizer": {"class": "AdamWConfig", "lr": 1e-05, "weight_decay": 0.0001, "grad_clip_norm": 10.0, "betas": [0.9, 0.999], "eps": 1e-08}, "scheduler": null, "torch_optimizer": "AdamW", "grad_clip_norm": 10.0, "cudnn_benchmark": true, "allow_tf32": true, "chunk_size": 100, "seed": 1000}`
  params total 51,613,582, trainable 51,613,582
- smolvla: `{"train_batch_size_default": 8, "mixed_precision": "no", "gradient_accumulation_steps": 1, "policy_use_amp_field": false, "use_policy_training_preset": true, "optimizer": {"class": "AdamWConfig", "lr": 0.0001, "weight_decay": 1e-10, "grad_clip_norm": 10.0, "betas": [0.9, 0.95], "eps": 1e-08}, "scheduler": {"class": "CosineDecayWithWarmupSchedulerConfig", "num_warmup_steps": 1000, "num_decay_steps": 30000, "peak_lr": 0.0001, "decay_lr": 2.5e-06}, "torch_optimizer": "AdamW", "grad_clip_norm": 10.0, "cudnn_benchmark": true, "allow_tf32": true, "chunk_size": 50, "seed": 1000}`
  params total 450,046,176, trainable 99,880,992

## Failed runs

- smolvla libero bs64: skipped_budget elapsed 915s > budget 720.0s
- smolvla hqf bs64: skipped_budget elapsed 915s > budget 720.0s
