# YOLO-GTGP: Ground Penetrating Radar Tunnel Geological Prediction
project for Unfavorable geological anomalies identification and location of Tunnel Geological Prediction for ground penetrating radar based on YOLO-GTGP

Cheng Chen, Xiao Tao, Shuang Luo, Deshan Feng, Li He, Wenxiu Yan, Weiliang Cao, Xun Wang,"Unfavorable geological anomalies identification and location of Tunnel Geological Prediction for ground penetrating radar based on YOLO-GTGP," Underground Space

# dataset
https://drive.google.com/file/d/15EmmMNdbmOEIo0oFNTRMXhUTe5P5ZkTb/view?usp=sharing
**Data Availability Statement:**
The simulated dataset utilized in this study is available from the corresponding author upon reasonable request for non-commercial academic purposes. However, the field-measured datasets involve proprietary engineering data and sensitive geological information from an actual tunnel project currently under construction. Due to strict confidentiality agreements with the collaborating engineering entities, the field data cannot be made publicly available at this time.

# Environment Integration (Crucial Step):
To ensure the ultralytics framework recognizes our custom modules, you must register them in your local environment:

Import GhostRepLite and DualPath_CNNTransformer into [Your_Env_Path]/site-packages/ultralytics/nn/tasks.py.

(Optional) For Soft-NMS integration, replace the standard NMS call in ultralytics/utils/ops.py with our implementation from models/yolo_gtgp_nms.py.
