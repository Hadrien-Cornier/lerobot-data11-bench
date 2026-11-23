# Read #5377’s depth recordings with the #5388 follow-up

The 16 original recording-and-seek tests passed on the [writer PR](https://github.com/huggingface/lerobot/pull/5377) at `e45dc16c`. Then the [index-free streaming reader](https://github.com/Hadrien-Cornier/lerobot/commit/e3060aaa5) consumed the 16 saved datasets. All 304 outputs matched the known input raster after quantization and dequantization. Each episode/frame ID appeared once in its dataset. These are synthetic depth recordings on Apple silicon, Python 3.12, CPU and PyAV. Both ordinary and streaming writers, two quantizers, two episode-length sequences and one or two encoder threads are represented. The writer PR’s own suite also tests individual random seeks. The follow-up reader check tests finite streamed output. Existing third-party recordings were outside this test.

With a benchmark environment containing PyAV, pytest, the LeRobot training dependencies and Datasets at `464fe790`, use independent source checkouts so each stage imports its intended implementation. Use a fresh scratch directory for the pytest fixtures.

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
fixtures=$(mktemp -d)
(cd "$WRITER_CHECKOUT" && PYTHONPATH="$WRITER_CHECKOUT/src" python -m pytest -q tests/datasets/test_depth_video_random_access.py --basetemp="$fixtures/depth")
PYTHONPATH="$READER_CHECKOUT/src" python review5388/correctness/check_depth_writer_fixture.py "$fixtures/depth"
```

[Frame counts for every saved dataset](measurements/depth-writer-5377-reader-5388.json)
