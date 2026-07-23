# TN-MJEPA

Publication code for source-disjoint self-supervised representation learning of in-train noise. The repository contains code and protocol descriptions only. It does not include the dataset, checkpoints, labels, caches, logs, figures, or reported result files.

## Repository layout

```text
pretrain/       data preparation, model definition, checkpoint compatibility, and pre-training
evaluation/     paper-level intrinsic evaluation and frozen weak-label probes
downstream/     supervised line-attribute fine-tuning
configs/        replaceable experiment protocols; no sample data
scripts/        reproducible multi-run commands
tests/          unit and protocol regression tests
```

## Model and metadata contract

The signal inputs are `tf_log`, aligned `pa_resampled`, and aligned `vel_resampled`. Velocity conditions the time-frequency and pressure encoders through FiLM, attention bias, and 16-group frequency gating. The three signal streams are fused with a metadata token and trained using EMA-target JEPA, overlap-aware multi-positive contrastive learning, representation regularization, and auxiliary regression.

The four-value tensor assembled by the loader has a deliberately separated contract:

| Columns | Fields | Role |
|---|---|---|
| 0-1 | `rms_pa_db`, `mean_vel` | normalized auxiliary targets and evaluation labels only |
| 2-3 | `duration_s`, `slice_type` | the only metadata values encoded and fused with signal features |

Changing columns 0-1 cannot change the metadata token. This behavior is covered by `tests/test_metadata_contract.py`.

## Data format

Each MATLAB sample must contain:

```text
tf_log          float array, shape (F, T)
pa_resampled    float array, shape (T,)
vel_resampled   float array, shape (T,)
freq_axis       float array, shape (F,)
time_axis       float array, shape (T,)
rms_pa_db       scalar auxiliary target
mean_vel        scalar auxiliary target
sampleMeta      sourceTag, sliceType, mileageStart_km,
                mileageEnd_km, and duration_s
```

`pa_raw` is not read. Variable-duration samples are retained; batching uses masks and padding. A user-supplied metadata index is needed only to construct a source-disjoint split:

```json
{
  "records": [
    {
      "name": "sample_00001.mat",
      "source": "run_or_section_id",
      "slice_type": "mileage",
      "duration_s": 0.84,
      "n_samples": 17203
    }
  ]
}
```

`name` and `source` are sufficient for split construction. The remaining fields allow the evaluation scripts to reuse the same index without reopening every file during dataset discovery. They describe samples but are not additional model inputs.

No source name is hard-coded in the model or generic training entry point. Other datasets can provide their own protocol JSON with the same `source_split.train/val/test` structure.

## Paper protocol

The exact paper settings are recorded in `configs/paper_protocol.json`, separately from the reusable code:

| Partition | Sources | Samples |
|---|---:|---:|
| Train | four source runs/sections listed in the protocol | 16,038 |
| Validation | `6号线-上行-草房-潞城` | 2,586 |
| Test | `6号线-下行-草房-潞城` | 2,514 |

Full models are trained for 60 epochs with seeds 42, 43, and 44. The evaluation seed is 20260719. The protocol runner rejects count or source drift before launching training.

## Environment and tests

Python 3.10 or newer is recommended.

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Reproduce pre-training

First compute training-only normalization statistics and the frozen split manifest. The dataset itself remains outside the repository.

```powershell
python -m pretrain.scripts.compute_train_stats `
  --dataset-dir path/to/sswpt_dataset `
  --output path/to/train_stats.json `
  --metadata-cache path/to/dataset_metadata.json `
  --protocol configs/paper_protocol.json
```

Then launch the three full-model runs. `--dry-run` prints the exact commands without training.

```powershell
python -m scripts.run_paper_pretraining `
  --dataset-dir path/to/sswpt_dataset `
  --stats-path path/to/train_stats.json `
  --output-root runs/paper
```

The full Section 5 evaluator also requires the five seed-42 ablations. They use the same 60-epoch budget and can be added without changing the default three-seed command:

```powershell
python -m scripts.run_paper_pretraining `
  --dataset-dir path/to/sswpt_dataset `
  --stats-path path/to/train_stats.json `
  --output-root runs/paper `
  --include-ablations
```

For another dataset, either use `pretrain.train_pretrain` directly or replace the source IDs and expected counts in a copied protocol file.

## Paper-level evaluation

The intrinsic evaluator operates on the complete frozen test partition and the eight Section 5 checkpoints. It writes global cross-view Retrieval@1 and margins, sample-wise JEPA cosine values, clean covariance spectra/effective rank, cross-modal linear CKA, training trajectories, SGA frequency-gate statistics, fusion attention enrichment, and pooling enrichment. Legacy checkpoint handling remains strict through the shared compatibility layer.

```powershell
python -m evaluation.paper_intrinsic `
  --experiment-root runs/paper `
  --dataset-dir path/to/sswpt_dataset `
  --stats-path path/to/train_stats.json `
  --output-root results/paper_intrinsic
```

Frozen weak-label probes are separate from pre-training:

```powershell
python -m evaluation.weak_label_probes `
  --checkpoint-root runs/paper `
  --dataset-dir path/to/sswpt_dataset `
  --stats-path path/to/train_stats.json `
  --metadata-cache path/to/dataset_metadata.json `
  --output-dir results/weak_label_probes
```

`mean_vel`, `rms_pa_db`, and `sliceType` in this script are probe targets, not additional model metadata. The slice-type input is masked for its corresponding probe.

The appendix intervention analysis is also outside the pre-training package. It uses the three Full checkpoints and a user-generated metadata index:

```powershell
python -m evaluation.interventions `
  --experiment-root runs/paper `
  --stats-path path/to/train_stats.json `
  --dataset-dir path/to/sswpt_dataset `
  --metadata-cache path/to/dataset_metadata.json `
  --output-dir results/interventions
```

## Downstream fine-tuning

Supervised evaluation is implemented in `downstream/finetune.py`. Its CSV manifest must provide `sample_id`, `file_name`, `source_file`, `slice_type`, `mileage_start_km`, `mileage_end_km`, `track_class`, `curve_radius_m`, and `corrugation`. Labels and manifests are not distributed with the code.

```powershell
python -m downstream.finetune `
  --manifest path/to/line_label_manifest.csv `
  --dataset-dir path/to/sswpt_dataset `
  --checkpoint runs/paper/full_seed42/best.pt `
  --stats-path path/to/train_stats.json `
  --epochs 20 `
  --split-strategy spatial `
  --representation fusion `
  --out-dir results/downstream_fusion
```

The same entry point supports `tf`, `wave`, `vel`, and `fusion` representations. Spatial splitting is the default; random splitting is retained only as an explicit comparison.

## Checkpoint compatibility

Current checkpoints load strictly. Earlier paper checkpoints may contain unused `target_fusion.*` and `target_meta_mlp.*` parameters from a superseded EMA target layout. Only these two prefixes are accepted and reported; missing weights or any other unexpected key raise an error.

## License

The code is released under the MIT License. See `LICENSE`.
