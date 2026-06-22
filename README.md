# YOLO-GTGP: Ground Penetrating Radar Tunnel Geological Prediction

Project for Unfavorable geological anomalies identification and location of Tunnel Geological Prediction for ground penetrating radar based on YOLO-GTGP

Cheng Chen, Xiao Tao, Shuang Luo, Deshan Feng, Li He, Wenxiu Yan, Weiliang Cao, Xun Wang, "Unfavorable geological anomalies identification and location of Tunnel Geological Prediction for ground penetrating radar based on YOLO-GTGP," *Underground Space*

## Dataset

https://drive.google.com/file/d/1GKYdNMy1Z8kZSNu_j_yn0UWIDfd4SXSq/view?usp=drive_link

**Data Availability Statement:**
The simulated dataset utilized in this study is available from the corresponding author upon reasonable request for non-commercial academic purposes. However, the field-measured datasets involve proprietary engineering data and sensitive geological information from an actual tunnel project currently under construction. Due to strict confidentiality agreements with the collaborating engineering entities, the field data cannot be made publicly available at this time.

## Pretrained Weights (.pt)

https://drive.google.com/file/d/1FAVRP-pKhP9bsaiUZW572mybUQleJcpf/view?usp=drive_link

---

## Environment Setup & Custom Module Integration

Follow the steps below to integrate the custom network modules into the Ultralytics YOLO framework.

### 1. Environment Download and Preparation

Please visit the official Ultralytics GitHub repository to download the source code of **v8.3.0 or later**:

```
https://github.com/ultralytics/ultralytics.git
```

### 2. Add Custom Module File

Place the `gtgp_blocks.py` file (containing all component implementations) directly into the following directory:

```
ultralytics/nn/modules/
```

### 3. Module Registration and Global Exposure

To enable the YOLO framework to globally recognize and invoke the newly added modules, modify `ultralytics/nn/modules/__init__.py` as follows:

#### Import modules

Add the following line to the import section at the top of the file:

```python
from .gtgp_blocks import GhostRepLite, DualPath_CNNTransformer
```

#### Update the `__all__` list

Append the following entries to the `__all__` tuple at the end of the file:

```python
"GhostRepLite",
"DualPath_CNNTransformer",
```

### 4. Parser Logic Adaptation

To ensure the model configuration file can correctly parse custom modules, modify the `parse_model` function in `ultralytics/nn/tasks.py`:

#### Import modules

Add the following line at the top of `tasks.py`:

```python
from ultralytics.nn.modules.gtgp_blocks import GhostRepLite, DualPath_CNNTransformer
```

#### Configure channel parsing logic

Inside the `parse_model` function, locate the `if` conditional statement that includes `Classify, Conv, ...`, then add the custom modules to the condition:

```python
if m in (..., GhostRepLite, DualPath_CNNTransformer):
    c1, c2 = ch[f], args[0]
```

#### Configure module depth (Repeats) parsing logic

Inside the `parse_model` function, locate the `if` branch with `args.insert(2, n)`, then add the custom modules to the condition:

```python
if m in (..., GhostRepLite, DualPath_CNNTransformer):
    args.insert(2, n)
    n = 1
```

## Custom Soft-NMS Integration ##

This section describes how to replace the native NMS with the provided enhanced Soft-NMS implementation that supports multiple IoU metrics.

### 1. Add the Soft-NMS Module ### 

Place the `gtgp_nms.py` file into the following directory of the Ultralytics project:

```
ultralytics/utils/
```

### 2. Replace Native NMS ### 

In the inference or evaluation script (e.g., `ultralytics/utils/ops.py`), locate the `non_max_suppression` function, and replace the native NMS call with the custom Soft-NMS.

Original native NMS code:

```python
import torch
i = torchvision.ops.nms(boxes, scores, iou_thres)
```

Replace with the enhanced Soft-NMS:

import torch
from ultralytics.utils.gtgp_nms import soft_nms

```python
i = soft_nms(
    bboxes=boxes,
    scores=scores,
    iou_thresh=0.45,        # IoU threshold for triggering Gaussian penalty
    sigma=0.5,              # Variance of the Gaussian penalty function
    score_threshold=0.25,   # Minimum confidence score to retain a bounding box
    iou_type='eiou'         # Supported metrics: iou / giou / diou / ciou / eiou / siou
)
```
