import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from prettytable import PrettyTable
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info

# ==============================================================================
# YOLO-GTGP Validation & Evaluation Script
# Objective: Evaluates the model on the GPR dataset and exports standardized 
#            metrics (mAP, Precision, Recall, FPS, etc.) formatted for manuscript.
# ==============================================================================

def get_weight_size(path):
    """Calculate the size of the model weight file in MB."""
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

if __name__ == '__main__':
    # Path to the trained YOLO-GTGP optimal weights
    model_path = 'runs/train/YOLO-GTGP-Train/weights/best.pt'
    model = YOLO(model_path) 
    
    # Execute validation process
    result = model.val(
        data='data/data.yaml',
        split='val',             
        imgsz=640,
        batch=16,
        # iou=0.6,               
        # rect=False,
        project='runs/val',
        name='YOLO-GTGP-Evaluation',
    )
    
    if model.task == 'detect':   # Applicable only for object detection tasks
        model_names = list(result.names.values())
        preprocess_time_per_image = result.speed['preprocess']
        inference_time_per_image = result.speed['inference']
        postprocess_time_per_image = result.speed['postprocess']
        all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image
        
        # Extract FLOPs and Parameters
        n_l, n_p, n_g, flops = model_info(model.model)
        
        print('-'*20 + ' Use the following results for the manuscript ' + '-'*20)

        # ---------------------------------------------------------
        # Table 1: Model Computational Information & Efficiency
        # ---------------------------------------------------------
        model_info_table = PrettyTable()
        model_info_table.title = "YOLO-GTGP Model Efficiency Info"
        model_info_table.field_names = [
            "GFLOPs", "Parameters", "Preprocess Time/Img", "Inference Time/Img", 
            "Postprocess Time/Img", "FPS (Overall)", "FPS (Inference Only)", "Weight Size"
        ]
        model_info_table.add_row([
            f'{flops:.1f}', f'{n_p:,}', 
            f'{preprocess_time_per_image / 1000:.6f}s', 
            f'{inference_time_per_image / 1000:.6f}s', 
            f'{postprocess_time_per_image / 1000:.6f}s', 
            f'{1000 / all_time_per_image:.2f}', 
            f'{1000 / inference_time_per_image:.2f}', 
            f'{get_weight_size(model_path)}MB'
        ])
        print(model_info_table)

        # ---------------------------------------------------------
        # Table 2: Model Detection Metrics
        # ---------------------------------------------------------
        model_metrice_table = PrettyTable()
        model_metrice_table.title = "YOLO-GTGP Detection Metrics"
        model_metrice_table.field_names = [
            "Class Name", "Precision (P)", "Recall (R)", "F1-Score", "mAP@50", "mAP@75", "mAP@50-95"
        ]
        
        # Populate class-specific metrics
        for idx, cls_name in enumerate(model_names):
            model_metrice_table.add_row([
                cls_name, 
                f"{result.box.p[idx]:.4f}", 
                f"{result.box.r[idx]:.4f}", 
                f"{result.box.f1[idx]:.4f}", 
                f"{result.box.ap50[idx]:.4f}", 
                f"{result.box.all_ap[idx, 5]:.4f}", # Thresholds: 50 55 60 65 70 75 80 85 90 95 
                f"{result.box.ap[idx]:.4f}"
            ])
            
        # Populate averaged overall metrics
        model_metrice_table.add_row([
            "All (Average)", 
            f"{result.results_dict['metrics/precision(B)']:.4f}", 
            f"{result.results_dict['metrics/recall(B)']:.4f}", 
            f"{np.mean(result.box.f1):.4f}", 
            f"{result.results_dict['metrics/mAP50(B)']:.4f}", 
            f"{np.mean(result.box.all_ap[:, 5]):.4f}", # Mean mAP@75
            f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
        ])
        print(model_metrice_table)

        # Export tables to a text file for direct copy-pasting into manuscript
        paper_data_path = result.save_dir / 'paper_data.txt'
        with open(paper_data_path, 'w+') as f:
            f.write(str(model_info_table))
            f.write('\n\n')
            f.write(str(model_metrice_table))
        
        print('-'*20, f' Results saved to {paper_data_path}... ', '-'*20)