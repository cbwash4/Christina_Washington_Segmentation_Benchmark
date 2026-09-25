# Image Segmentation Benchmarking

**Course:** CCIS 727 Computer Vision  
**Institution:** Clark Atlanta University  
**Department:** Cyber Physical Systems  
**Author:** Christina Washington  

---

## 1. Project Overview

This project implements a reusable image segmentation benchmarking
framework for comparing traditional, convolutional, instance-based,
and Transformer-based segmentation methods under a controlled
experimental protocol.

The benchmark evaluates segmentation quality together with model
complexity and computational efficiency.

The required methods are:

1. K-Means traditional segmentation baseline
2. FCN-ResNet50
3. U-Net
4. U-Net++
5. SegNet
6. DeepLabV3-ResNet50
7. PSPNet
8. Mask R-CNN
9. YOLO Segmentation
10. SegFormer-B0

The benchmark contains separate semantic-segmentation and
instance-segmentation evaluation tracks because semantic mIoU and
instance mask AP measure different segmentation tasks and are not
treated as directly interchangeable metrics.

---

## 2. Dataset

The benchmark uses a reproducible subset of the COCO 2017 dataset.

### Foreground Classes

| Benchmark ID | Class | COCO Category ID |
|---:|---|---:|
| 0 | Background | N/A |
| 1 | Person | 1 |
| 2 | Car | 3 |
| 3 | Bicycle | 2 |
| 4 | Dog | 18 |
| 5 | Cat | 17 |

### Dataset Partitions

| Partition | Images |
|---|---:|
| Training | 5,000 |
| Validation | 1,000 |
| Testing | 1,000 |
| **Total** | **7,000** |

Random seed: **42**

The exact image IDs for every partition are stored in:

    data/manifests/train_image_ids.json
    data/manifests/val_image_ids.json
    data/manifests/test_image_ids.json
    data/manifests/dataset_split_manifest.json

The 7,000-image candidate subset was created using deterministic
class-balanced selection. Multilabel stratification was then used
to preserve target-class representation across the training,
validation, and testing partitions. A deterministic seeded
adjustment produced the exact 5,000 / 1,000 / 1,000 partition sizes.

No image ID is shared between the training, validation, and testing
partitions.

### Current Class Representation

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| Person | 3,563 | 722 | 710 |
| Car | 1,156 | 235 | 231 |
| Bicycle | 850 | 172 | 169 |
| Dog | 843 | 170 | 169 |
| Cat | 852 | 171 | 170 |

---

## 3. Segmentation Tasks

### Semantic Segmentation

The semantic benchmark predicts one class ID for every valid pixel.

Semantic models:

- FCN-ResNet50
- U-Net
- U-Net++
- SegNet
- DeepLabV3-ResNet50
- PSPNet
- SegFormer-B0

Semantic masks use the following benchmark class mapping:

    0 = Background
    1 = Person
    2 = Car
    3 = Bicycle
    4 = Dog
    5 = Cat

The final mask-generation procedure documents the treatment of
overlapping annotations, crowd regions, non-selected classes, and
ignored pixels.

### Instance Segmentation

The instance benchmark retains separate object masks, class labels,
bounding boxes, and instance identities.

Instance models:

- Mask R-CNN with ResNet50-FPN
- YOLO11n-seg

Semantic and instance segmentation quality results are reported
separately.

### Traditional Segmentation

K-Means color segmentation is included as the required
non-deep-learning baseline.

K-Means cluster IDs are not semantic class labels. Therefore,
K-Means is used primarily for qualitative comparison unless a
semantic class-mapping procedure learned exclusively from training
data is implemented. Test labels are not used to assign cluster
identities.

---

## 4. Experimental Configuration

The common benchmark configuration is:

| Setting | Value |
|---|---|
| Input resolution | 512 x 512 |
| Training budget | Minimum 25 epochs |
| Batch size | 8 |
| Optimizer | AdamW |
| Initial learning rate | 0.0001 |
| Random seed | 42 |
| Semantic classes | 6 |
| Hardware | NVIDIA A100-SXM4-40GB |

Mixed precision may be used and will be documented with the final
runs.

Architecture-specific deviations, preprocessing requirements,
effective batch sizes, and optimization settings will be recorded
with the corresponding experiment.

For semantic segmentation, the common starting loss is:

    Loss = 0.5 * CrossEntropyLoss + 0.5 * DiceLoss

The final implementation documents the Dice formulation, class
averaging convention, absent-class handling, and ignore-label
policy.

---

## 5. Model Configurations

### 5.1 FCN-ResNet50

The FCN implementation uses a ResNet50 backbone for fully
convolutional dense prediction. Classification outputs are replaced
with segmentation heads appropriate for the six benchmark semantic
classes.

### 5.2 U-Net

U-Net uses an encoder, bottleneck, and decoder with skip connections
between corresponding encoder and decoder stages. The current
implementation uses a ResNet34 encoder.

### 5.3 U-Net++

U-Net++ uses a ResNet34 encoder with nested and dense skip pathways.
The architecture is included for direct comparison with standard
U-Net.

### 5.4 SegNet

SegNet uses an encoder-decoder architecture in which max-pooling
indices from the encoder are retained and used for decoder
unpooling. Unlike U-Net, the decoder does not rely on transferring
the full corresponding encoder feature maps through skip
connections.

### 5.5 DeepLabV3-ResNet50

DeepLabV3 uses a ResNet50 backbone, atrous convolution, and Atrous
Spatial Pyramid Pooling (ASPP) to capture contextual information at
multiple receptive-field scales.

### 5.6 PSPNet

PSPNet uses pyramid pooling to combine local features with broader
image-level contextual information. The current implementation uses
a ResNet34 backbone.

### 5.7 Mask R-CNN

Mask R-CNN uses a ResNet50 backbone with a Feature Pyramid Network
(FPN), Region Proposal Network (RPN), RoI Align, classification,
bounding-box regression, and instance-mask prediction.

### 5.8 YOLO Segmentation

The selected YOLO instance-segmentation model is:

    Variant: YOLO11n-seg
    Checkpoint: yolo11n-seg.pt
    Package: ultralytics

The final benchmark records the exact package version, training
configuration, input size, confidence settings, IoU settings, and
inference configuration.

### 5.9 SegFormer-B0

SegFormer-B0 provides the Transformer-based semantic segmentation
comparison. It uses a hierarchical Transformer encoder and a
lightweight MLP decoder.

The current pretrained initialization source is:

    nvidia/segformer-b0-finetuned-ade-512-512

The output head is adapted for the six benchmark semantic classes.

### 5.10 K-Means

K-Means color clustering is the traditional computer-vision
baseline. The current implementation uses six clusters and
random_state=42.

---

## 6. Preprocessing and Augmentation

All semantic models use a common target resolution of 512 x 512
unless an architecture-specific requirement is documented.

Training preprocessing may include:

- Resize
- Horizontal flip
- Crop
- Rotation
- Color jitter
- Tensor conversion
- Normalization

Validation and testing use deterministic preprocessing.

All geometric transformations must be applied identically to the
image and corresponding segmentation masks. Instance bounding boxes
must also be updated when image geometry changes.

Discrete segmentation masks use label-preserving interpolation such
as nearest-neighbor resizing.

Color transformations are applied to images only.

The exact final transform order, probabilities, and parameters will
be recorded before benchmark training begins.

---

## 7. Evaluation Metrics

### Semantic Segmentation Metrics

The semantic track reports:

- Pixel accuracy
- Per-class IoU
- Mean IoU (mIoU)
- Dice / pixel F1
- Pixel precision
- Pixel recall
- Boundary precision
- Boundary recall
- Boundary F1
- Small-object performance

The final evaluation configuration declares whether background is
included in macro averages, how absent classes are handled, and
which pixels are ignored.

### Instance Segmentation Metrics

Mask R-CNN and YOLO Segmentation are evaluated using:

- Mask AP
- AP50
- AP75
- Average Recall
- Optional box AP

The primary COCO-style mask AP is averaged over IoU thresholds from
0.50 through 0.95 in increments of 0.05.

The evaluator, score filtering, class averaging, IoU thresholds,
and maximum detections are recorded with the final results.

### Efficiency Metrics

The benchmark measures:

- Total parameters
- Trainable parameters
- Saved model size
- Total training time
- Average time per epoch
- Inference latency
- Images per second
- Peak GPU memory

Inference benchmarking uses evaluation mode without gradient
tracking. CUDA timing includes synchronization around timed regions,
and warm-up iterations are performed before measurements.

Batch-one latency and batched throughput are reported separately
where applicable.

---

## 8. Project Structure

    Christina_Washington_Segmentation_Benchmark/
    |
    |-- data/
    |   `-- manifests/
    |-- annotations/
    |-- models/
    |-- checkpoints/
    |-- results/
    |-- masks/
    |-- predictions/
    |-- plots/
    |-- logs/
    |-- confusion_matrices/
    |-- configs/
    |
    |-- src/
    |   |-- dataset.py
    |   |-- mask_utils.py
    |   |-- models.py
    |   |-- losses.py
    |   |-- train.py
    |   |-- evaluate.py
    |   |-- metrics.py
    |   |-- visualize.py
    |   `-- benchmark.py
    |
    |-- run_benchmark.py
    |-- requirements.txt
    |-- README.md
    `-- configuration.yaml

Semantic and instance models use separate adapters where their
targets, outputs, losses, training procedures, or evaluators are
incompatible.

---

## 9. Benchmark Execution

The completed framework will support individual models, complete
semantic or instance tracks, and the full benchmark.

### Individual Semantic Model

    python run_benchmark.py --task semantic --model unet

    python run_benchmark.py --task semantic --model deeplabv3

### Complete Semantic Track

    python run_benchmark.py --task semantic --model all

### Mask R-CNN

    python run_benchmark.py --task instance --model maskrcnn

### YOLO Segmentation

    python run_benchmark.py --task instance --model yolo_seg

### Complete Benchmark

    python run_benchmark.py --task all --model all

The completed command-line interface will define the behavior of
each command, required prerequisites, checkpoint handling, and
evaluation-only execution.

Evaluation will be repeatable from saved checkpoints without
requiring every model to be retrained.

---

## 10. Generated Outputs

### Results

The benchmark automatically generates:

    results/
        semantic_segmentation_results.csv
        instance_segmentation_results.csv
        segmentation_efficiency_results.csv

Experimental measurements are generated from saved training and
evaluation outputs rather than manually entered.

### Required Figures

The completed benchmark generates:

1. Semantic model vs. mIoU
2. Semantic model vs. Dice score
3. Model vs. parameter count
4. Model vs. saved model size
5. Model vs. training time
6. Model vs. inference images per second
7. Semantic mIoU vs. parameter count
8. Semantic mIoU vs. inference latency
9. Per-class IoU across semantic models
10. Mask R-CNN vs. YOLO Segmentation: mask AP, AP50, and AP75

Generated figures are stored in:

    plots/

Prediction examples are stored by model under:

    predictions/

Confusion matrices are stored under:

    confusion_matrices/

Checkpoints are stored under:

    checkpoints/

Training and evaluation logs are stored under:

    logs/

---

## 11. Reproducibility

The benchmark uses random seed 42 and identical image partitions for
all architectures.

The exact training, validation, and testing image IDs are preserved
in the dataset manifests.

Training checkpoints are selected using validation data only.

Semantic checkpoints are selected using validation mIoU.

Instance checkpoints are selected using a declared validation
instance metric such as mask AP.

The held-out test partition is reserved for final evaluation and is
not used for checkpoint or hyperparameter selection.

Per-run configurations, class mappings, split manifests, training
histories, raw evaluation outputs, software versions, input
resolution, batch size, numeric precision, hardware information, and
run identifiers are retained with benchmark measurements.

---

## 12. Project Status

This repository is being developed for:

**CCIS 727 Computer Vision - Practical Assignment #4**

Completed framework components currently include:

- Project directory structure
- Master experiment configuration
- Reproducible COCO image-ID partitions
- Semantic Dice and combined loss implementation
- Semantic segmentation metrics
- Reusable semantic training interface
- Semantic model factory
- FCN-ResNet50 implementation
- U-Net implementation
- U-Net++ implementation
- SegNet implementation
- DeepLabV3-ResNet50 implementation
- PSPNet implementation
- SegFormer-B0 implementation
- Mask R-CNN initialization and interface verification
- YOLO11n-seg initialization and interface verification
- K-Means baseline implementation

Remaining components include dataset mask generation, final
segmentation-aware transformations, complete training runs, instance
evaluation, boundary evaluation, small-object evaluation, efficiency
benchmarking, automatic result tables, required plots, final
command-line execution, and qualitative prediction analysis.

Final experimental measurements will be added only after the
corresponding benchmark runs have been completed.
