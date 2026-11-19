# Check the two LeRobot commits against the original reader

The three revisions are stock #5388 at `1a60cb4b800c2947834342752c32f064c8bf143c`, decoder reuse at `992364e125736f5fefb4c2bf4690c0b3d187b951`, and grouped RGB reads at `adbdf5f0218898b187b4d20376a4bdee4a97d388`. All three checks below exercise the installed `StreamingLeRobotDataset` through a PyTorch DataLoader.

The three native revisions produced identical RGB, state, action and ordered episode/frame hashes on all 1,000 MolmoAct2 batches on the same Mac. The compact fixture also preserved every field, batch boundary and partial batch in all four loader configurations. [MolmoAct2 results](measurements/native-molmo-parity.json) · [Finite fixture results](measurements/native-grouping-fixture.json)

## Reproduce frame and order checks

Run from the benchmark's `review5388/` directory in the [pinned dataset environment](method.md#reproduce-a-cloud-reader-measurement). Set `PY` to its Python interpreter. The environment setup script puts the pinned Datasets code at commit `464fe790` in that environment. Keep the interpreter, TorchCodec and FFmpeg versions fixed across the comparisons. The public fixture requires FFmpeg on PATH.

```bash
PY="$WORK/venv/bin/python"
checkdir=$(mktemp -d)
git clone --branch perf/5388-grouped-rgb \
    https://github.com/Hadrien-Cornier/lerobot.git "$checkdir/lerobot"
for spec in stock:1a60cb4b800c2947834342752c32f064c8bf143c \
            reuse:992364e125736f5fefb4c2bf4690c0b3d187b951 \
            grouped:adbdf5f0218898b187b4d20376a4bdee4a97d388; do
    git -C "$checkdir/lerobot" worktree add --detach "$checkdir/${spec%%:*}" "${spec#*:}"
done
PYTHONPATH="$checkdir/stock/src" "$PY" correctness/fixture.py "$checkdir/fixture"
for mode in stock reuse grouped; do
    reference=()
    if [[ "$mode" != stock ]]; then reference=(--reference "$checkdir/stock.json"); fi
    PYTHONPATH="$checkdir/$mode/src" "$PY" correctness/check_native_grouping.py \
        "$checkdir/fixture" "$checkdir/$mode.json" --ffmpeg "${reference[@]}"
done
```

The finite fixture has 786 examples, three camera views and packed videos with nonzero episode offsets. The consumer batch size is 19, so some batches cross the reader's internal eight-sample decode groups. This check compares every current RGB frame to an independent FFmpeg decode and every structured value to the source Parquet row. Repeating with `--workers 4`, `--float`, or `--history` compares each mode to its own stock reference. For camera history, omit `--ffmpeg` and compare all returned frames and masks with stock.

## Replay the 1,000 MolmoAct2 batches

The following uses the real bucket and the saved [ordered episode/frame IDs](internvideo_reference.json.gz). It checks all 8,000 examples and hashes every returned RGB byte, state and action in order. The script checks the bucket inventory before and after each run. A fresh stock capture gives the reference on your decoder stack. Local stock RGB hashes differed between the Mac and Linux environments used in this review.

```bash
PYTHONPATH="$checkdir/stock/src" "$PY" check_native_molmo.py --write-stock "$checkdir/molmo-stock.json"
for mode in reuse grouped; do
    PYTHONPATH="$checkdir/$mode/src" "$PY" check_native_molmo.py --against "$checkdir/molmo-stock.json"
done
```

These commands can share disk caches for equivalence checks. The [earlier controlled benchmark](internvideo.md) used fresh application caches and a forward and reverse run order to measure the prototype.
