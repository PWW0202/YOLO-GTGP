# ==========================================
# 🚀 Independent Evaluation Subsystem II: Soft-NMS IoU Variant Ablation
# Supported Models: Soft-NMS(GIoU/DIoU/CIoU/EIoU/SIoU)
# ==========================================
import os
import torch
import torchvision
from ultralytics import YOLO

# Import custom Soft-NMS core algorithm
from ultralytics.utils.gpgt_nms import soft_nms 

# ==========================================
# ⚙️ Experiment Configuration & Hyperparameter Matrix
# ==========================================
MODEL_NAME = "Soft-NMS(EIoU)"
WEIGHT_PATH = "weights/Soft-NMS(EIoU).pt"
DATA_YAML = "dataset.yaml"

# 🌟 Soft-NMS Baseline Configuration
IOU_TYPE = "eiou"        # Available pool: giou, diou, ciou, eiou, siou
SIGMA = 0.5
SCORE_THRESH = 0.15      

# ==========================================
# 🔧 Dynamic NMS Hijacking Engine (Monkey Patching)
# ==========================================
original_nms = torchvision.ops.nms

def custom_soft_nms_wrapper(boxes, scores, iou_threshold):
    """Intercepts native NMS requests and forces routing to Soft-NMS with specific IoU penalties."""
    return soft_nms(
        bboxes=boxes,
        scores=scores,
        iou_thresh=iou_threshold, 
        sigma=SIGMA,
        score_threshold=SCORE_THRESH,
        iou_type=IOU_TYPE
    )

# 💉 Inject core engine
torchvision.ops.nms = custom_soft_nms_wrapper

# ==========================================
# 🏁 Core Evaluation Routine
# ==========================================
def main():
    print("\n" + "="*60)
    print(f"🔄 Initiating Variant Ablation Test | Target Model: {MODEL_NAME}")
    print(f"🧠 Hijack Config -> IoU Type: {IOU_TYPE.upper()} | Truncation Threshold: {SCORE_THRESH}")
    print("="*60)
    
    if not os.path.exists(WEIGHT_PATH):
        print(f"❌ Fatal Error: Weight file not found -> {WEIGHT_PATH}")
        return

    try:
        model = YOLO(WEIGHT_PATH)
        metrics = model.val(
            data=DATA_YAML,
            split="test",
            batch=256,
            device=0,           
            workers=0,          
            save_txt=True,      
            save_conf=True      
        )
        
        p, r = metrics.box.mp, metrics.box.mr
        map50, map95 = metrics.box.map50, metrics.box.map
        
        print("\n" + "#"*50)
        print(f"📈 [{MODEL_NAME}] Dynamic Hijacking Test Results Report")
        print("#"*50)
        print(f" 🔹 Precision (P) : {p:.4f}")
        print(f" 🔹 Recall (R)    : {r:.4f}")
        print(f" 🔹 mAP@50        : {map50:.4f}")
        print(f" 🔹 mAP@50-95     : {map95:.4f}")
        print("-" * 50 + "\n")
        
    except Exception as e:
        print(f"❌ Evaluation crashed: {e}")
    finally:
        # 🧹 VRAM Protection Mechanism
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()