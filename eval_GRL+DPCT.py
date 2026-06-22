import os
import torch
from ultralytics import YOLO

# ==========================================
# ⚙️ Experiment Configuration
# ==========================================
MODEL_NAME = "YOLO-GRL-DPCT"
WEIGHT_PATH = "weights/GRL+DPCT.pt"
DATA_YAML = "dataset.yaml"

# ==========================================
# 🏁 Core Evaluation Routine
# ==========================================
def main():
    print("\n" + "="*60)
    print(f"🔄 Initiating Standard Benchmark | Target Model: {MODEL_NAME}")
    print("="*60)
    
    if not os.path.exists(WEIGHT_PATH):
        print(f"❌ Fatal Error: Weight file not found -> {WEIGHT_PATH}")
        return

    try:
        # 📦 Instantiate the official model (Clean version, no NMS hijacking)
        model = YOLO(WEIGHT_PATH)
        
        # 🏃 Execute high-throughput evaluation
        metrics = model.val(
            data=DATA_YAML,
            split="test",
            batch=256,
            device=0,           
            workers=0,          
            save_txt=True,      
            save_conf=True      
        )
        
        # 📊 Extract core metrics
        p, r = metrics.box.mp, metrics.box.mr
        map50, map95 = metrics.box.map50, metrics.box.map
        
        print("\n" + "#"*50)
        print(f"📈 [{MODEL_NAME}] Clean Benchmark Results Report")
        print("#"*50)
        print(f" 🔹 Precision (P) : {p:.4f}")
        print(f" 🔹 Recall (R)    : {r:.4f}")
        print(f" 🔹 mAP@50        : {map50:.4f}")
        print(f" 🔹 mAP@50-95     : {map95:.4f}")
        print("-" * 50 + "\n")
        
    except Exception as e:
        print(f"❌ Evaluation crashed: {e}")
    finally:
        # 🧹 VRAM Protection Mechanism: Force garbage collection to prevent OOM
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()