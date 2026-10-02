| Dataset | Files | Arm | Files/s | vs BASE (95% CI) | vs PATCH (95% CI) | KiB/file | Round trips/file | Full build |
|---|---|---|---|---|---|---|---|---|
| BitRobot/HIW-500-LeRobot | 11,873 | BASE | 2.8 | 1 | 0.12 (0.11 to 0.13) | 32,332 | 2.15 | 69.9 min |
| BitRobot/HIW-500-LeRobot | 11,873 | PATCH | 22.7 | 8.03 (7.65 to 8.85) | 1 | 527 | 1.23 | 8.7 min |
| BitRobot/HIW-500-LeRobot | 11,873 | A1M | 16.5 | 5.84 (5.51 to 6.54) | 0.71 (0.69 to 0.75) | 1,536 | 1.00 | 12.0 min |
| BitRobot/HIW-500-LeRobot | 11,873 | A3M | 13.9 | 4.99 (4.53 to 5.47) | 0.60 (0.59 to 0.62) | 3,584 | 1.00 | 14.2 min |
| BitRobot/HIW-500-LeRobot | 11,873 | A1M-c32 | 17.7 | 7.06 (6.34 to 7.66) | 0.81 (0.76 to 0.94) | 1,536 | 1.00 | 11.2 min |
| allenai/MolmoAct2-BimanualYAM-Dataset | 5,860 | BASE | 6.9 | 1 | 0.48 (0.45 to 0.56) | 9,469 | 1.25 | 14.2 min |
| allenai/MolmoAct2-BimanualYAM-Dataset | 5,860 | PATCH | 14.4 | 2.10 (1.80 to 2.23) | 1 | 518 | 1.07 | 6.8 min |
| allenai/MolmoAct2-BimanualYAM-Dataset | 5,860 | A1M | 10.9 | 1.67 (1.50 to 1.74) | 0.78 (0.76 to 0.82) | 1,537 | 1.01 | 8.9 min |
| allenai/MolmoAct2-BimanualYAM-Dataset | 5,860 | A3M | 8.9 | 1.35 (1.25 to 1.45) | 0.69 (0.64 to 0.73) | 3,585 | 1.01 | 10.9 min |
| allenai/MolmoAct2-BimanualYAM-Dataset | 5,860 | A1M-c32 | 12.7 | 1.76 (1.68 to 1.93) | 0.90 (0.81 to 0.95) | 1,537 | 1.01 | 7.7 min |
| cadene/agibot_alpha_v30 | 4,072 | BASE | 0.8 | 1 | 0.08 (0.08 to 0.09) | 126,443 | 5.97 | 82.9 min |
| cadene/agibot_alpha_v30 | 4,072 | PATCH | 9.8 | 12.05 (11.07 to 12.84) | 1 | 945 | 2.00 | 6.9 min |
| cadene/agibot_alpha_v30 | 4,072 | A1M | 9.2 | 11.36 (10.62 to 12.16) | 0.93 (0.91 to 0.99) | 1,725 | 1.14 | 7.4 min |
| cadene/agibot_alpha_v30 | 4,072 | A3M | 7.9 | 9.36 (9.07 to 10.36) | 0.80 (0.76 to 0.85) | 3,586 | 1.00 | 8.6 min |
| cadene/agibot_alpha_v30 | 4,072 | A1M-c32 | 9.6 | 11.88 (10.35 to 12.69) | 0.98 (0.87 to 1.03) | 1,722 | 1.14 | 7.0 min |
| lerobot/aloha_sim_insertion_human | 1 | BASE | 0.5 | 1 | 0.13 (0.13 to 0.13) | 127,125 | 6.00 | 1.9 s |
| lerobot/aloha_sim_insertion_human | 1 | PATCH | 3.9 | 7.59 (7.45 to 7.76) | 1 | 661 | 2.00 | 0.2 s |
| lerobot/aloha_sim_insertion_human | 1 | A1M | 4.9 | 9.63 (8.69 to 10.07) | 1.21 (1.13 to 1.34) | 1,536 | 1.00 | 0.1 s |
| lerobot/aloha_sim_insertion_human | 1 | A3M | 3.3 | 6.94 (5.80 to 8.40) | 0.85 (0.72 to 1.08) | 3,584 | 1.00 | 0.2 s |
| lerobot/aloha_sim_insertion_human | 1 | A1M-c32 | 4.9 | 9.93 (9.15 to 10.49) | 1.25 (1.19 to 1.38) | 1,536 | 1.00 | 0.1 s |
| lerobot/aloha_sim_insertion_scripted | 1 | BASE | 0.5 | 1 | 0.13 (0.13 to 0.14) | 127,095 | 6.00 | 1.8 s |
| lerobot/aloha_sim_insertion_scripted | 1 | PATCH | 4.0 | 7.54 (6.96 to 7.79) | 1 | 631 | 2.00 | 0.2 s |
| lerobot/aloha_sim_insertion_scripted | 1 | A1M | 4.6 | 8.84 (8.09 to 9.22) | 1.22 (1.12 to 1.26) | 1,536 | 1.00 | 0.1 s |
| lerobot/aloha_sim_insertion_scripted | 1 | A3M | 4.1 | 7.87 (7.41 to 8.35) | 1.06 (1.02 to 1.09) | 3,584 | 1.00 | 0.2 s |
| lerobot/aloha_sim_insertion_scripted | 1 | A1M-c32 | 4.7 | 8.80 (7.99 to 9.28) | 1.19 (1.06 to 1.28) | 1,536 | 1.00 | 0.1 s |
| lerobot/aloha_sim_transfer_cube_human | 1 | BASE | 0.5 | 1 | 0.14 (0.14 to 0.16) | 127,095 | 6.00 | 1.8 s |
| lerobot/aloha_sim_transfer_cube_human | 1 | PATCH | 3.6 | 6.98 (6.44 to 7.33) | 1 | 631 | 2.00 | 0.2 s |
| lerobot/aloha_sim_transfer_cube_human | 1 | A1M | 4.2 | 7.74 (6.96 to 8.12) | 1.10 (0.98 to 1.25) | 1,536 | 1.00 | 0.2 s |
| lerobot/aloha_sim_transfer_cube_human | 1 | A3M | 4.1 | 7.87 (7.18 to 8.58) | 1.14 (0.99 to 1.28) | 3,584 | 1.00 | 0.2 s |
| lerobot/aloha_sim_transfer_cube_human | 1 | A1M-c32 | 4.4 | 8.30 (7.45 to 9.25) | 1.19 (1.03 to 1.31) | 1,536 | 1.00 | 0.1 s |
| lerobot/droid_1.0.1 | 813 | BASE | 4.5 | 1 | 0.68 (0.66 to 0.73) | 8,564 | 1.18 | 3.0 min |
| lerobot/droid_1.0.1 | 813 | PATCH | 6.6 | 1.48 (1.37 to 1.51) | 1 | 674 | 1.92 | 2.1 min |
| lerobot/droid_1.0.1 | 813 | A1M | 6.2 | 1.37 (1.27 to 1.42) | 0.94 (0.90 to 0.96) | 1,638 | 1.85 | 2.2 min |
| lerobot/droid_1.0.1 | 813 | A3M | 5.4 | 1.18 (1.15 to 1.25) | 0.80 (0.79 to 0.85) | 3,688 | 1.84 | 2.5 min |
| lerobot/droid_1.0.1 | 813 | A1M-c32 | 6.2 | 1.36 (1.26 to 1.49) | 0.94 (0.91 to 0.95) | 1,651 | 1.86 | 2.2 min |
| lerobot/pusht | 1 | BASE | 2.4 | 1 | 0.67 (0.65 to 0.82) | 10,825 | 2.00 | 0.3 s |
| lerobot/pusht | 1 | PATCH | 3.4 | 1.50 (1.22 to 1.53) | 1 | 663 | 2.00 | 0.2 s |
| lerobot/pusht | 1 | A1M | 4.2 | 1.71 (1.64 to 1.83) | 1.19 (1.10 to 1.51) | 1,536 | 1.00 | 0.2 s |
| lerobot/pusht | 1 | A3M | 4.2 | 1.74 (1.55 to 1.84) | 1.27 (1.05 to 1.38) | 3,584 | 1.00 | 0.2 s |
| lerobot/pusht | 1 | A1M-c32 | 4.2 | 1.72 (1.39 to 1.89) | 1.24 (0.98 to 1.39) | 1,536 | 1.00 | 0.2 s |
| yaak-ai/L2D | 9,624 | BASE | 16.5 | 1 | 0.66 (0.63 to 0.68) | 4,096 | 1.00 | 9.7 min |
| yaak-ai/L2D | 9,624 | PATCH | 25.5 | 1.50 (1.46 to 1.59) | 1 | 517 | 1.00 | 6.3 min |
| yaak-ai/L2D | 9,624 | A1M | 15.8 | 0.97 (0.94 to 1.01) | 0.61 (0.60 to 0.66) | 1,541 | 1.00 | 10.1 min |
| yaak-ai/L2D | 9,624 | A3M | 13.7 | 0.82 (0.65 to 0.88) | 0.54 (0.40 to 0.55) | 3,588 | 1.00 | 11.7 min |
| yaak-ai/L2D | 9,624 | A1M-c32 | 18.8 | 1.14 (1.07 to 1.20) | 0.75 (0.70 to 0.78) | 1,541 | 1.00 | 8.5 min |
| zekaiwang/trex_dataset | 7,844 | BASE | 13.2 | 1 | 0.74 (0.69 to 0.82) | 4,597 | 1.02 | 9.9 min |
| zekaiwang/trex_dataset | 7,844 | PATCH | 17.7 | 1.34 (1.22 to 1.46) | 1 | 543 | 1.02 | 7.4 min |
| zekaiwang/trex_dataset | 7,844 | A1M | 11.4 | 0.86 (0.83 to 0.95) | 0.64 (0.62 to 0.72) | 1,552 | 1.01 | 11.5 min |
| zekaiwang/trex_dataset | 7,844 | A3M | 10.9 | 0.83 (0.80 to 0.85) | 0.61 (0.59 to 0.64) | 3,600 | 1.01 | 12.0 min |
| zekaiwang/trex_dataset | 7,844 | A1M-c32 | 16.5 | 1.26 (1.19 to 1.39) | 0.93 (0.91 to 0.98) | 1,557 | 1.01 | 7.9 min |

Total full build, all datasets: BASE 3.2 h, PATCH 38.2 min, A1M 52.1 min, A3M 59.8 min, A1M-c32 44.6 min
Equality (same index in BASE, PATCH, A1M): BitRobot/HIW-500-LeRobot 32/32, allenai/MolmoAct2-BimanualYAM-Dataset 32/32, cadene/agibot_alpha_v30 32/32, lerobot/aloha_sim_insertion_human 4/4, lerobot/aloha_sim_insertion_scripted 4/4, lerobot/aloha_sim_transfer_cube_human 4/4, lerobot/droid_1.0.1 32/32, lerobot/pusht 4/4, yaak-ai/L2D 32/32, zekaiwang/trex_dataset 32/32
Failed runs: 0
