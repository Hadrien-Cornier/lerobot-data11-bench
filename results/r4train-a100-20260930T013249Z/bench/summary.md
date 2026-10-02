# DATA-11 r4train: training step throughput with synthetic batches

- lerobot_sha: e0d50211ef236143ae867228662b7dfaba554f02
- python: 3.12.3
- host: j-hadriencornier-6abc66bc031314b696343423-f7jsg11a-5bca0-nwpqp
- warmup: 10
- steps: 50
- device: cuda
- gpu: NVIDIA A100-SXM4-80GB
- torch: 2.11.0+cu128 cuda 12.8 cudnn 91900

update = forward + backward + grad clip + optimizer + scheduler (`lerobot_train.update_policy`).
prep = uint8->float on CPU + policy preprocessor (tokenizer, device copy, normalization).
samples/s = batch / median update. Camera frames/s = samples/s x cameras.

| policy | setup | cams | batch | status | update median ms | update p90 ms | prep median ms | samples/s | cam frames/s | samples/s incl prep | peak alloc GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| act | abc | 3 | 8 | ok | 49.3 | 50.5 | 6.9 | 162.3 | 486.8 | 141.8 | 1.57 |
| act | droid | 3 | 8 | ok | 48.4 | 49.6 | 7.6 | 165.2 | 495.6 | 142.0 | 1.59 |
| act | molmo | 3 | 8 | ok | 76.2 | 76.6 | 18.9 | 105.0 | 315.1 | 84.0 | 4.36 |
| act | libero | 2 | 8 | ok | 41.3 | 50.3 | 3.7 | 193.8 | 387.5 | 176.3 | 1.60 |
| act | hqf | 3 | 8 | ok | 232.5 | 233.1 | 166.0 | 34.4 | 103.2 | 20.1 | 18.99 |
| smolvla | abc | 3 | 8 | ok | 173.4 | 189.0 | 5.2 | 46.1 | 138.4 | 44.4 | 3.56 |
| smolvla | droid | 3 | 8 | ok | 166.7 | 171.6 | 5.7 | 48.0 | 144.0 | 46.2 | 3.56 |
| smolvla | molmo | 3 | 8 | ok | 184.6 | 186.7 | 26.6 | 43.3 | 130.0 | 38.3 | 3.61 |
| smolvla | libero | 2 | 8 | ok | 165.6 | 171.4 | 4.8 | 48.3 | 96.6 | 46.9 | 2.91 |
| smolvla | hqf | 3 | 8 | ok | 165.3 | 167.7 | 102.0 | 48.4 | 145.2 | 29.9 | 3.74 |
| smolvla | abc | 3 | 64 | ok | 639.4 | 640.9 | 64.0 | 100.1 | 300.3 | 91.0 | 19.31 |
| smolvla | droid | 3 | 64 | ok | 639.2 | 640.7 | 75.3 | 100.1 | 300.4 | 89.5 | 19.32 |
| smolvla | molmo | 3 | 64 | ok | 635.9 | 637.3 | 363.2 | 100.6 | 301.9 | 64.0 | 19.69 |
| smolvla | libero | 2 | 64 | skipped_budget | - | - | - | - | - | - | - |
| smolvla | hqf | 3 | 64 | skipped_budget | - | - | - | - | - | - | - |

## Defaults found (per policy)

- act: `{"train_batch_size_default": 8, "mixed_precision": "no", "gradient_accumulation_steps": 1, "policy_use_amp_field": false, "use_policy_training_preset": true, "optimizer": {"class": "AdamWConfig", "lr": 1e-05, "weight_decay": 0.0001, "grad_clip_norm": 10.0, "betas": [0.9, 0.999], "eps": 1e-08}, "scheduler": null, "torch_optimizer": "AdamW", "grad_clip_norm": 10.0, "cudnn_benchmark": true, "allow_tf32": true, "chunk_size": 100, "seed": 1000}`
  params total 51,613,582, trainable 51,613,582
- smolvla: `{"train_batch_size_default": 8, "mixed_precision": "no", "gradient_accumulation_steps": 1, "policy_use_amp_field": false, "use_policy_training_preset": true, "optimizer": {"class": "AdamWConfig", "lr": 0.0001, "weight_decay": 1e-10, "grad_clip_norm": 10.0, "betas": [0.9, 0.95], "eps": 1e-08}, "scheduler": {"class": "CosineDecayWithWarmupSchedulerConfig", "num_warmup_steps": 1000, "num_decay_steps": 30000, "peak_lr": 0.0001, "decay_lr": 2.5e-06}, "torch_optimizer": "AdamW", "grad_clip_norm": 10.0, "cudnn_benchmark": true, "allow_tf32": true, "chunk_size": 50, "seed": 1000}`
  params total 450,046,176, trainable 99,880,992

## Failed runs

- smolvla libero bs64: skipped_budget elapsed 482s > budget 420.0s
- smolvla hqf bs64: skipped_budget elapsed 482s > budget 420.0s
