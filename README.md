---

# YOLO-GTGP: Ground Penetrating Radar Tunnel Geological Prediction#

This repository contains the official implementation, dataset split, and pretrained weights for the paper:

**"Unfavorable geological anomalies identification and location of Tunnel Geological Prediction for ground penetrating radar based on YOLO-GTGP"** submitted to *Underground Space*.

**Authors:** Cheng Chen, Xiao Tao, Shuang Luo, Deshan Feng, Li He, Wenxiu Yan, Weiliang Cao, Xun Wang

---

## 1. Data and Weights Download

To independently verify our reported results, please download the necessary resources below:

### Dataset

* **Download Link:** [Google Drive Link to Dataset](https://drive.google.com/file/d/1GKYdNMy1Z8kZSNu_j_yn0UWIDfd4SXSq/view?usp=drive_link)
* **Description:** Contains the test split of the simulated GPR dataset with corresponding annotations.
* **Usage:** Extract the dataset and place it in the `datasets/` directory of your project root.

> **Data Availability Statement:** > The simulated dataset utilized in this study is available from the corresponding author upon reasonable request for non-commercial academic purposes. However, the field-measured datasets involve proprietary engineering data and sensitive geological information from an actual tunnel project currently under construction. Due to strict confidentiality agreements with the collaborating engineering entities, the field data cannot be made publicly available at this time.

### Pretrained Weights (.pt)

* **Download Link:** [Google Drive Link to Weights](https://drive.google.com/file/d/1FAVRP-pKhP9bsaiUZW572mybUQleJcpf/view?usp=drive_link)
* **Description:** Includes the trained model weights for the full YOLO-GTGP model and its ablation variants.
* **Usage:** Place the `.pt` files directly into the `weights/` directory.

---

## 2. Environment Setup & Custom Module Integration

Follow the steps below to integrate our custom network modules into the Ultralytics YOLO framework.

### Step 2.1: Clone Ultralytics YOLO

Download the source code of **v8.3.0 or later** from the official repository and install requirements:

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip install -r requirements.txt

```

### Step 2.2: Add Custom Module Files

Place the provided `gtgp_blocks.py` file (containing `GhostRepLite` and `DualPath_CNNTransformer`) directly into the following directory:

`ultralytics/nn/modules/`

### Step 2.3: Register Modules Globally

Modify the `ultralytics/nn/modules/__init__.py` file to expose the new modules globally.

**Import modules:** Add this line at the top of the file:

```python
from .gtgp_blocks import GhostRepLite, DualPath_CNNTransformer

```

**Update the `__all__` list:** Append the module names to the `__all__` tuple at the bottom of the file:

```python
__all__ = (
    # ... existing modules
    "GhostRepLite",
    "DualPath_CNNTransformer",
)

```

### Step 2.4: Adapt Parser Logic

Modify the `parse_model` function in `ultralytics/nn/tasks.py` so the custom YAML configuration file can be parsed correctly.

**Import modules:** Add this line at the top of `tasks.py`:

```python
from ultralytics.nn.modules.gtgp_blocks import GhostRepLite, DualPath_CNNTransformer

```

**Configure channel parsing:** Locate the large `if m in (...)` block inside the `parse_model` function and append our custom modules:

```python
if m in (
    Classify,
    Conv,
    # ... existing modules
    GhostRepLite, 
    DualPath_CNNTransformer
):
    c1, c2 = ch[f], args[0]

```

**Configure module depth (Repeats):** Locate the depth parsing branch further down in the same function and append our custom modules:

```python
if m in (
    C2f,
    C2fAttn,
    # ... existing modules
    GhostRepLite, 
    DualPath_CNNTransformer
):
    args.insert(2, n)
    n = 1

```

---

## 3. Custom Soft-NMS Integration

To replace the native NMS with our enhanced Soft-NMS (supporting multiple IoU metrics), follow these instructions:

### Step 3.1: Add the Soft-NMS File

Place the provided `gtgp_nms.py` file directly into the following directory:

`ultralytics/utils/`

### Step 3.2: Replace Native NMS in Code

Open `ultralytics/utils/ops.py`, locate the `non_max_suppression` function, and swap the native NMS call with our custom implementation.

**Original Code:**

```python
import torchvision

# ...
i = torchvision.ops.nms(boxes, scores, iou_thres)

```

**Replace with:**

```python
from ultralytics.utils.gtgp_nms import soft_nms

# ...
i = soft_nms(
    bboxes=boxes,
    scores=scores,
    iou_thresh=0.45,       
    sigma=0.5,             
    score_threshold=0.25,  
    iou_type='eiou'         # Supported metrics: iou / giou / diou / ciou / eiou / siou
)

```

---
