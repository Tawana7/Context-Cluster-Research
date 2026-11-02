# Causal Structure Learning vs. Hand-Designed Graphs for Micro-Expression Recognition

Code for my BSc Honours research report (University of the Witwatersrand). The project compares
three conditions inside a CI-MER style causal intervention framework on CASME II, evaluated with
leave-one-subject-out (LOSO) cross-validation:

| Condition | `--graph_mode` | What it is |
|---|---|---|
| No graph | `none` | Plain CI-Block on the backbone feature |
| Hand-designed graph | `handdesigned` | FACS anatomy based AU graph, fixed across folds |
| FCI-learned graph | `fci` | AU graph learned with FCI, refit per fold on that fold's training subjects only |

This README explains how to run the code.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── scripts/                      all Python code (listed in the script reference below)
├── slurm/                        the Slurm job scripts used on the cluster
└── results/                      small result CSVs from the reported runs
```

The repository does **not** contain the CASME II images, the optical flow images, or the CoC-Tiny
checkpoint. See "Data and files you must supply".

## 1. Installation

Python 3.10 was used. Create an environment and install the requirements:

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux or macOS
pip install -r requirements.txt
```

Training also needs the **Context-Cluster** repository, which provides the CoC-Tiny backbone
(`models` package) and is not copied here:

```bash
git clone https://github.com/ma-xu/Context-Cluster.git
```

`train_cimer_casme2.py` runs `import models`, so Python must be able to find that repository.
There are two ways to do this.

**Option A (recommended): put it on the Python path.** Do this once per terminal session.

```powershell
# Windows PowerShell
$env:PYTHONPATH = "C:\path\to\Context-Cluster"
```
```bash
# Linux
export PYTHONPATH=$HOME/Context-Cluster:$PYTHONPATH
```

**Option B: copy the files in.** Copy everything in `scripts/` into the root of your Context-Cluster
clone and run the commands from there. This is how the cluster experiments were run.

The other scripts (flow, analysis, figures) do not need Context-Cluster.

## 2. Data and files you must supply

| Item | Where to get it | Notes |
|---|---|---|
| CASME II frames (cropped) | The dataset authors, under their licence agreement | Expected layout: `<frames_root>/subNN/<Filename>/reg_img<K>.jpg` |
| CASME II coding sheet | Same download, `CASME2-coding-20140508.xlsx` | Needs the columns `Subject`, `Filename`, `OnsetFrame`, `ApexFrame`, `OffsetFrame`, `Action Units`, `Estimated Emotion` |
| CoC-Tiny ImageNet checkpoint | Linked in the Context-Cluster README | Optional but used for all reported results. Pass it with `--pretrained` |

CASME II is distributed under a licence, so no images or flow images are included in this repository.

## 3. Running the pipeline

All commands assume you are in the repository root. Replace the paths with your own. If a path
contains spaces or special characters, keep the quotes. The first check below catches path problems.

### Step 1: Check that your paths work

```bash
python scripts/diagnose_path.py --frames_root "C:\path\to\CASME2\Cropped"
```

It prints whether the folder exists and, if not, what the parent folder actually contains.

### Step 2: Describe the dataset

```bash
python scripts/analyze_casme2.py --xlsx "C:\path\to\CASME2-coding-20140508.xlsx" --images "C:\path\to\CASME2\Cropped" --outdir dataset_report
```

Writes `casme2_report.md`, `casme2_class_distribution.csv` and `casme2_subject_distribution.csv`.
These supply the class counts, subject imbalance, sequence length statistics and the AU frequency table.
`--images` is optional and cross-checks the folders against the sheet.

### Step 3: Compute optical flow

```bash
python scripts/compute_flow_casme2.py --xlsx "C:\path\to\CASME2-coding-20140508.xlsx" --frames_root "C:\path\to\CASME2\Cropped" --out_root "C:\path\to\CASME2\Flow" --method tvl1
```

Two colour-coded flow images are written for each sequence (`flow_onset_apex.png` and
`flow_apex_offset.png`). Sequences whose annotated frames are missing on disk are skipped and
recorded in `flow_skip_log.csv`. Use `--clamp_to_last_frame` to keep them instead, but note that this
changes which frames are used, so only do so deliberately. `--method farneback` needs only
`opencv-python`, and `--method raft` needs `torch` and `torchvision`. The reported results use `tvl1`,
which needs `opencv-contrib-python`.

### Step 4 (optional): Look at the flow

```bash
python scripts/sanity_check_flow.py --xlsx "...xlsx" --frames_root "...\Cropped" --subject 1 --filename EP02_01f --out sanity_check.png
```

Saves the raw frames, a raw difference image and the flow side by side, so you can judge the flow quality.

### Step 5 (optional): Inspect the AU vocabulary and the graphs

```bash
python scripts/au_features.py --xlsx "...xlsx" --min_count 2
python scripts/hand_designed_au_graph.py --xlsx "...xlsx" --min_count 2
python scripts/fci_graph_learning.py --xlsx "...xlsx" --alpha 0.05 --test chisq
```

The first prints the AU nodes used. The second prints how many of the 14 hand-designed edges are
usable with those nodes. The third runs FCI once on the whole dataset as a sanity check. This is
not a LOSO fold, so it is for inspection only.

### Step 6: Smoke test, then train the three conditions

Run a very small test first to confirm everything is wired up:

```bash
python scripts/train_cimer_casme2.py --data_root "C:\path\to\CASME2\Flow" --xlsx "...xlsx" \
    --pretrained checkpoints/model_best.pth.tar --graph_mode none \
    --max_folds 2 --epochs 2 --results_csv smoke_none.csv
```

Then run each condition. Use the **same settings for all three**, changing only `--graph_mode` and
the output name. The placeholders below are the values listed in the report's Section 4.7. Copy them
from your own job scripts in `slurm/`.

```bash
python -u scripts/train_cimer_casme2.py \
    --data_root "C:\path\to\CASME2\Flow" \
    --xlsx "C:\path\to\CASME2-coding-20140508.xlsx" \
    --pretrained checkpoints/model_best.pth.tar \
    --graph_mode none \
    --epochs 50 --batch_size 8 --lr 1e-4 \
    --val_subjects <K> --val_min_samples <M> --patience <P> \
    --ckpt_dir ckpt --resume \
    --results_csv results_v2_none.csv
```

Repeat with `--graph_mode handdesigned --results_csv results_v2_handdesigned.csv` and
`--graph_mode fci --results_csv results_v2_fci.csv`. For `fci`, you can add
`--save_graphs_dir graphs_fci` to keep every fold's adjacency matrix.

Important behaviour to know about:

* **Data protocol per fold.** Test is the held-out subject. Validation is `--val_subjects` other subjects.
  Training is everyone else. Early stopping and checkpoint selection use validation only.
* **Resuming.** With `--ckpt_dir` and `--resume`, finished folds are skipped and an interrupted fold
  continues from `last.pt`. The same command can simply be run again after an interruption. A settings
  fingerprint stops you from resuming with changed hyperparameters, so use a different `--run_tag` if
  you want to change them.
* **Results files.** `--results_csv results_v2_none.csv` also creates `results_v2_none_epochs.csv`
  and `results_v2_none_predictions.csv`. Files from an older version with different columns are refused,
  so use a fresh name for a fresh run.
* **GPU.** The script uses CUDA if it finds it. Add `--require_gpu` on a cluster to stop instead of
  silently training on CPU if the GPU is missing or faulty.
* **Weights and Biases.** Add `--use_wandb --wandb_project <name>`. On a cluster without internet,
  set `WANDB_MODE=offline` and run `wandb sync` from the login node later.
* **Other options.** `--early_stop_metric val_f1`, `--scheduler cosine`, `--label_smoothing`,
  `--backbone_lr_mult`, `--class_weighted` and `--weight_decay` exist but were left at their defaults
  for the reported results.

### Step 7: Summarise the results

```bash
python scripts/summarize_results.py --results_dir . --prefix results_v2_
```

Prints mean and standard deviation over folds, pooled accuracy, UAR and UF1, and early stopping
statistics for each condition. It also writes `results_v2_summary_table.csv`.

### Step 8: Statistical analysis and tables

```bash
python scripts/analyze_results_extra.py --results_dir . --prefix results_v2_ --out_dir report_tables
```

Writes paired Wilcoxon tests, subject-level bootstrap intervals, per-class recall, confusion matrices
and mean epoch curves into `report_tables/`.

### Step 9: Figures and LaTeX tables

```bash
python scripts/make_report_figures.py --tables_dir report_tables --out_dir figures
```

Writes `confusion_matrices.pdf`, `training_curves.pdf` and three `.tex` tables.

### Step 10 (optional): Report-only figures

```bash
python scripts/make_flow_figure.py --xlsx "...xlsx" --frames_root "...\Cropped" \
    --flow_root "...\Flow" --subject <N> --filename <sequence> --out flow_example
python scripts/make_overview_diagram.py
```

The first makes the onset, apex, offset and flow figure. Check the CASME II licence before putting
any participant's face in a document. The second writes the pipeline overview as
`pipeline_overview.drawio`, `.png` and `.pdf`.

## 4. Running on a Slurm cluster

The reported results were produced on the Wits mscluster. A template for one job is below. Adjust the
partition, memory, module name and paths for your own cluster.

```bash
#!/bin/bash
#SBATCH --job-name=cimer_none
#SBATCH --partition=bigbatch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load anaconda/anaconda-2025-12-2
export PYTHONPATH=$HOME/Context-Cluster

apptainer exec --nv "$ANACONDA_SIF" bash -c "source /opt/anaconda/etc/profile.d/conda.sh && conda activate cimer && cd \$HOME/<repo> && python -u scripts/train_cimer_casme2.py --data_root \$HOME/Context-Cluster/Flow --xlsx \$HOME/CASME_II/CASME2-coding-20140508.xlsx --pretrained \$HOME/Context-Cluster/checkpoints/model_best.pth.tar --graph_mode none --require_gpu --val_subjects <K> --patience <P> --ckpt_dir \$HOME/ckpt --resume --results_csv results_v2_none.csv"
```

Notes from running on this cluster:

* Compute nodes have no internet. Download the checkpoint and install packages from the login node.
* Anaconda is containerised, so `conda activate` only works inside the `apptainer exec` line above.
* Partition names can mislead. `bigbatch` had GPUs, and `stampede` was CPU only. Check the device line
  in the log (`Using device: cuda (...)`).
* After a node failure, resubmit the same script. `--resume` continues where it stopped.

## 5. Output files

| File | Written by | Contents |
|---|---|---|
| `flow_skip_log.csv` | `compute_flow_casme2.py` | Sequences skipped or clamped, with reasons |
| `results_<tag>.csv` | `train_cimer_casme2.py` | One row per fold |
| `results_<tag>_epochs.csv` | `train_cimer_casme2.py` | One row per fold and epoch |
| `results_<tag>_predictions.csv` | `train_cimer_casme2.py` | One row per test sequence |
| `ckpt/<tag>/fold_XX/` | `train_cimer_casme2.py` | `best.pt`, `last.pt`, `result.json` |
| `report_tables/*.csv` | `analyze_results_extra.py` | Tests, intervals, per-class recall, curves |
| `figures/*` | `make_report_figures.py` | Figures and `.tex` tables used in the report |

## 6. Script reference

| Script | Purpose |
|---|---|
| `analyze_casme2.py` | Dataset statistics from the coding sheet |
| `diagnose_path.py` | Checks that a dataset path reaches Python correctly |
| `compute_flow_casme2.py` | TV-L1, Farneback or RAFT flow for onset to apex and apex to offset |
| `sanity_check_flow.py` | Side by side view of frames and flow for one sequence |
| `au_features.py` | Parses the AU column into binary vectors and finds the AU vocabulary |
| `hand_designed_au_graph.py` | The FACS based edge list and its adjacency matrix |
| `fci_graph_learning.py` | FCI graph learning, edge overlap and cross-fold stability |
| `ci_block.py` | The CI-Block (reimplemented from the CI-MER paper). Has a small self test: `python scripts/ci_block.py` |
| `graph_ci_block.py` | Graph-gated CI-Block used by the two graph conditions. Self test: `python scripts/graph_ci_block.py` |
| `train_cimer_casme2.py` | LOSO training and evaluation, early stopping, checkpoints |
| `summarize_results.py` | Headline numbers per condition |
| `analyze_results_extra.py` | Paired tests, bootstrap intervals, per-class recall, curves |
| `make_report_figures.py` | Confusion matrix and curve figures, LaTeX tables |
| `make_flow_figure.py` | Example sequence figure with flow |
| `make_overview_diagram.py` | Pipeline overview diagram |

## 7. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'models'` | Context-Cluster is not on the Python path. Set `PYTHONPATH` (Option A above). |
| `unrecognized arguments: ...` | An old copy of a script is being run. Copy the current file again and confirm with `grep` that the flag exists. |
| `module 'cv2' has no attribute 'optflow'` | Install `opencv-contrib-python`, not only `opencv-python`. Or use `--method farneback`. |
| `libGL.so.1: cannot open shared object file` | Install `opencv-python-headless` instead of `opencv-python` on a machine without a display. |
| Flow script says the frames are not found | Run `diagnose_path.py`. On Windows, non-ASCII folder names (such as the numeral in "CASME II") can break paths. The scripts read files in a Unicode-safe way, but quotes around the path are still needed. |
| `Using device: cpu` on a GPU node | The GPU is missing or faulty. Add `--require_gpu` so the job stops, then resubmit. |
| `already exists with different columns` | A results file from an older version. Use a new `--results_csv` name. |

Known quirk: `compute_flow_casme2.py` contains an unused first `main()` stub above the real one. It does
no harm, but it can be deleted.

## 8. Reproducing the reported results

The reported runs used one seed (`--seed 1`) per condition, TV-L1 flow, the CoC-Tiny ImageNet
checkpoint, 26 LOSO folds and the settings in Section 4.7 of the report. Training is not bit-for-bit
deterministic on GPU, so small differences from the reported numbers are possible even with identical
settings.

## 9. Acknowledgements and licence

The backbone comes from Context-Cluster (Ma et al., ICLR 2023). The CI-Block is my own reimplementation
from the equations in the CI-MER paper (Fang, Wu and Cao, ICME 2025), because no public code exists
for it. CASME II is the work of Yan et al. (2014) and is used under its own licence.
