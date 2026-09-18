# From SegNet to MedSAM: Benchmarking Brain Tumour Segmentation on T1-Weighted MRI

A controlled comparison of classical CNNs, prompt-based foundation models and
detection-first architectures for brain tumour segmentation, all evaluated on the
same 3,064-image T1-weighted contrast-enhanced MRI dataset under one preprocessing
and evaluation pipeline.

**Headline result:** a domain-adapted foundation model (MedSAM, two-phase transfer
learning) reached **Dice 0.82**, beating the best classical CNN baseline
(Attention U-Net, **0.805**) while keeping interactive prompt control. Off-the-shelf
SAM scored **0.13** on the same data — a **6.3×** gap that is the clearest measure
of the natural-image-to-medical-image domain shift in this study.

---

## Why this exists

Segmentation papers rarely compare architectures on equal footing: different
preprocessing, different splits, different metrics. Every model here sees the same
data, the same split and the same evaluation code, so the ranking means something.

The study spans three families that are usually benchmarked separately:

- **Classical CNNs** — SegNet, Attention U-Net
- **Prompt-based foundation models** — SAM (zero-shot), MedSAM (fine-tuned)
- **Detection-first** — LSM-YOLO, RCS-YOLO

## Results

### Segmentation (mean validation Dice)

| Model | Dice ↑ | Notes |
|---|---|---|
| SAM (zero-shot) | 0.130 | No fine-tuning; prompts derived from ground-truth masks |
| SegNet | 0.613 | Encoder–decoder baseline |
| MedSAM (pre-transfer) | 0.780 | Medical-domain pretrained, before our transfer regimen |
| Attention U-Net | 0.805 | Best classical CNN; spatial attention aids boundary delineation |
| **MedSAM (post-transfer)** | **0.820** | **Best overall** — two-phase transfer learning |

### Detection

| Model | mAP ↑ |
|---|---|
| LSM-YOLO / RCS-YOLO | up to 0.920 |

Confirms real-time tumour localisation is viable alongside dense segmentation.

### What the numbers say

Three findings, in order of how much they surprised me:

1. **Zero-shot SAM is unusable here (0.13).** Even with bounding-box and point
   prompts derived from ground truth — a generous setting — the natural-image prior
   does not transfer to MRI.
2. **Domain adaptation closes that gap entirely.** The same architecture, medically
   pretrained and fine-tuned, goes from 0.13 to 0.82.
3. **Attention matters more than depth among the CNNs.** Attention U-Net beat SegNet
   by 0.19 Dice, almost entirely on boundary quality.

The practical conclusion: a domain-adapted foundation model beats a purpose-built
CNN *and* keeps prompt flexibility, so you get accuracy without giving up
interactive control in a clinical workflow.

## Dataset

- **3,064** T1-weighted contrast-enhanced MRI slices
- **Classes:** meningioma, glioma, pituitary tumour
- **Source format:** MATLAB `.mat`
- **Split:** 80/20 train/validation (`sklearn.model_selection.train_test_split`)

### Preprocessing

Raw `.mat` slices → normalised, contrast-enhanced, three-channel inputs at
**1024×1024** for the SAM/MedSAM pipeline. CNN baselines train at **128×128** for
tractability. Augmentation is baked into the dataset conversion rather than applied
on the fly; batches are reshuffled each epoch.

A custom `BrainTumorDataset` (subclass of `torch.utils.data.Dataset`) handles
loading and conversion.

## Training

### CNN baselines

| Setting | Value |
|---|---|
| Loss | Compound Dice + binary cross-entropy |
| Optimiser | Adam, lr 1e-4 |
| Batch size | 16 |
| Epochs | 40 (converges by 20–30; callbacks usually stop early) |
| Early stopping | `val_dice_coef`, patience 10, restore best weights |
| LR schedule | ReduceLROnPlateau, factor 0.5, patience 5, min 1e-6 |
| Checkpointing | Best validation Dice |

### MedSAM transfer learning

Two-phase regimen:

1. **Freeze the prompt encoder** — preserves prompt semantics
2. **Enable gradient checkpointing in the vision transformer** — fits the ViT in
   available GPU memory
3. **Fine-tune the mask decoder** over successive four-epoch stages

Mixed-precision training throughout (`GradScaler`).

### Hardware

Kaggle P100 and Google Colab A100.

## Evaluation

Dice coefficient is the primary metric, with IoU and pixel accuracy as supporting
measures. Shared evaluation code scores every model identically. Training and
validation Dice/loss curves are plotted per epoch; validation predictions are
overlaid on the source images every 5 epochs for qualitative inspection.

## Repository layout

```
SegNet/            Attention UNet/     MedSAMv1/       MedSAMv2/
Naive SAM/         LSM-YOLO/           RCS-YOLO/
*.ipynb            Comparison and evaluation notebooks
```

## Limitations

- 2D slice-wise segmentation only; no 3D volumetric context
- SAM/MedSAM prompts are derived from ground-truth masks, which is an upper bound
  on what an interactive clinical workflow would achieve
- Single dataset, so cross-scanner and cross-protocol generalisation is untested
- CNN baselines train at 128×128 while MedSAM uses 1024×1024 — resolution is a
  confound in the CNN-vs-foundation-model comparison

## Context

BSc graduation thesis, Sichuan University (adviser: Yin Hao). This work leads into
current MSc research at Harbin Institute of Technology on *Conflict-Aware
Multimodal Learning for Trustworthy Chest X-ray Interpretation* — extending the
question from "which architecture segments best" to "what should a model do when
its modalities disagree".

## Author

**Bassel-Abdessamad Grine** — MSc Computer Science, Harbin Institute of Technology
[LinkedIn](https://www.linkedin.com/in/abdessamad-grine/) · [Portfolio](https://abdessamadgrine.vercel.app)
