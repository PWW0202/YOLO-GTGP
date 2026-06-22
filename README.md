# YOLO-GTGP: Ground Penetrating Radar Tunnel Geological Prediction
project for Unfavorable geological anomalies identification and location of Tunnel Geological Prediction for ground penetrating radar based on YOLO-GTGP

Cheng Chen, Xiao Tao, Shuang Luo, Deshan Feng, Li He, Wenxiu Yan, Weiliang Cao, Xun Wang,"Unfavorable geological anomalies identification and location of Tunnel Geological Prediction for ground penetrating radar based on YOLO-GTGP," Underground Space

# dataset
https://drive.google.com/file/d/1GKYdNMy1Z8kZSNu_j_yn0UWIDfd4SXSq/view?usp=drive_link
**Data Availability Statement:**
The simulated dataset utilized in this study is available from the corresponding author upon reasonable request for non-commercial academic purposes. However, the field-measured datasets involve proprietary engineering data and sensitive geological information from an actual tunnel project currently under construction. Due to strict confidentiality agreements with the collaborating engineering entities, the field data cannot be made publicly available at this time.

# .pt
https://drive.google.com/file/d/1FAVRP-pKhP9bsaiUZW572mybUQleJcpf/view?usp=drive_link

# Integration Guide: YOLO-GTGP Custom Modules
To integrate the GhostRepLite and DualPath_CNNTransformer modules into your Ultralytics YOLO (v8.3.0+) project, please follow these steps:

1. Environment Setup
Clone the official Ultralytics repository and install it in editable mode:

Bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip install -e .
2. Add Custom Module File
Place the gpgt_blocks.py file containing the module implementations directly into the ultralytics/nn/modules/ directory.

3. Module Registration
To ensure the YOLO framework recognizes the new modules, modify ultralytics/nn/modules/__init__.py:

Import the modules: Add the following line to the top of the file:

Python
from .gpgt_blocks import GhostRepLite, DualPath_CNNTransformer
Update __all__: Add the module names to the __all__ tuple at the end of the file:

Python
"GhostRepLite",
"DualPath_CNNTransformer",
4. Update Model Parser
To enable the model configuration files to parse these modules correctly, update ultralytics/nn/tasks.py:

Import the modules: Add to the top of the file:

Python
from ultralytics.nn.modules.gpgt_blocks import GhostRepLite, DualPath_CNNTransformer
Channel Parsing: In the parse_model function, locate the if statement containing Classify, Conv, ... and add your modules:

Python
if m in (..., GhostRepLite, DualPath_CNNTransformer):
    c1, c2 = ch[f], args[0]
Depth (Repeats) Parsing: In the same function, locate the if branch containing args.insert(2, n) and add your modules:

Python
if m in (..., GhostRepLite, DualPath_CNNTransformer):
    args.insert(2, n)
    n = 1
