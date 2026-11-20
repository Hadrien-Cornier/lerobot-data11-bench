# Does indexed #3917 catch patched #5388 on MolmoAct2?

## The first two pairs reached 30,000 batches

Patched #5388 delivered every one of the first 30,000 batches sooner on both hosts. The lead at the last batch was 9.49 seconds on 8 vCPUs and 216.58 seconds on 16 vCPUs. Each reader delivered 240,000 distinct usable examples. Its ordered episode and frame IDs matched across hosts. Each reader used its own sampling order.

| Host | Reader | Batch 1 | Batch 1,000 | Batch 30,000 | Examples/s, batches 126–30,000 | Projected epoch |
|---|---|---:|---:|---:|---:|---:|
| 8 vCPU | #3917 with matching index | 144.49 s | 235.99 s | 40 min 35.87 s | 104.85 | 8.40 days |
| 8 vCPU | Patched #5388 | 20.28 s | 116.21 s | 40 min 26.39 s | 100.82 | 8.73 days |
| 16 vCPU | #3917 with matching index | 133.20 s | 203.31 s | 36 min 36.34 s | 116.38 | 7.56 days |
| 16 vCPU | Patched #5388 | 24.43 s | 122.46 s | 32 min 59.76 s | 124.62 | 7.06 days |

On 8 vCPUs, the patched reader's lead grew to 275 seconds at batch 13,175 and shrank to 9.49 seconds by batch 30,000. Its rate fell from 125 examples/s before the first refill (starting after batch 125) to 89 examples/s after the second. On 16 vCPUs, those rates were 130 and 122 examples/s. Each host ran one pair and the reader order was reversed, so host size and order changed together. [8-vCPU gap at every batch](measurements/molmo-native-refills-r01.svg) · [16-vCPU gap at every batch](measurements/molmo-native-refills-xl02.svg) · [Combined measurements](measurements/molmo-native-refills-both.json)

The pinned MolmoAct2 source contains 76,046,658 examples. With eight examples per batch, a full pass has 9,505,832 full batches and one final batch of two. Each 30,000-batch run covers 0.316% of the dataset. The epoch estimate adds time already measured to the time needed for the remaining examples at the observed rate after batch 125. It estimates CPU input delivery without model computation.

## What the comparison includes

The clocks start before dataset construction. For #3917 they also include downloading and validating a matching 1.83 GB publisher-built index. We supplied the index from a separate repository to simulate publication because the bucket's existing indexes have an older schema. The receipt took 10.30 s on 8 vCPUs and 9.41 s on 16 vCPUs.

Both readers used three current RGB camera views, current state, one current action, a pool of 32 episodes, batch size eight, seed 42 and zero DataLoader workers. A verified refill requires every anchor from the previous pool cohort and at least one anchor from each replacement span to reach the consumer. The first two verified refills were at batches 18,557 and 29,163 for #3917, and 6,878 and 13,175 for patched #5388. The former averaged 5.91 distinct episodes per batch over the full 30,000, the latter 7.06. They performed this comparison at their native levels of diversity. The source manifest matched before and after every run. [Reader patch](https://github.com/lhoestq/lerobot/pull/1) · [Run the experiment](method.md#run-patched-5388-and-indexed-3917-through-two-pool-refills)
