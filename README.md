# YOLO-GTGP: Ground Penetrating Radar Tunnel Geological Prediction
project for Unfavorable geological anomalies identification and location of Tunnel Geological Prediction for ground penetrating radar based on YOLO-GTGP

Cheng Chen, Xiao Tao, Shuang Luo, Deshan Feng, Li He, Wenxiu Yan, Weiliang Cao, Xun Wang,"Unfavorable geological anomalies identification and location of Tunnel Geological Prediction for ground penetrating radar based on YOLO-GTGP," Underground Space

# dataset
https://drive.google.com/file/d/15EmmMNdbmOEIo0oFNTRMXhUTe5P5ZkTb/view?usp=sharing
**Data Availability Statement:**
The simulated dataset utilized in this study is available from the corresponding author upon reasonable request for non-commercial academic purposes. However, the field-measured datasets involve proprietary engineering data and sensitive geological information from an actual tunnel project currently under construction. Due to strict confidentiality agreements with the collaborating engineering entities, the field data cannot be made publicly available at this time.

# Environment Integration (Crucial Step):
Step 1: Add the Source FileCreate a new file named gpgt_block.py inside the ultralytics/nn/modules/ directory of your Ultralytics installation and paste the provided code into it.  Next, expose these classes by adding them to ultralytics/nn/modules/__init__.py:Python# Inside ultralytics/nn/modules/__init__.py
from .gpgt_block import GhostRepLite, DualPath_CNNTransformer

__all__ = (
    # ... existing modules ...
    "GhostRepLite",
    "DualPath_CNNTransformer",
)
Step 2: Register Modules in the YOLO ParserYOLO builds its models dynamically from YAML files using a parser. You must tell the parser how to read your new blocks.Open ultralytics/nn/tasks.py and locate the parse_model function.Import your blocks at the top of the file:Pythonfrom ultralytics.nn.modules.gpgt_block import GhostRepLite, DualPath_CNNTransformer
2. **Add them to the parsing loop:** Scroll down inside `parse_model` to where the standard modules (like `C2f`, `Bottleneck`) are evaluated, and add your modules:

   ```python
   # Inside the `for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):` loop in tasks.py
   
   if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
            BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3,
            GhostRepLite, DualPath_CNNTransformer): # <--- ADD YOUR BLOCKS HERE
       c1, c2 = ch[f], args[0]
       if c2 != no:  # if not output
           c2 = make_divisible(c2 * gw, 8)

       args = [c1, c2, *args[1:]]
       
       # Handle the specific depth multiplier (n) for your blocks
       if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3, 
                GhostRepLite, DualPath_CNNTransformer): # <--- ADD YOUR BLOCKS HERE
           args.insert(2, n)  # number of repeats
           n = 1
