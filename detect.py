from ultralytics import YOLO

# ==============================================================================
# YOLO-GTGP Inference & Detection Script
# ==============================================================================

def main():
    # Load the optimal YOLO-GTGP weights obtained from training
    model = YOLO("runs/train/YOLO-GTGP-Train/weights/best.pt")

    # Define the source directory containing unlabelled GPR B-scan images.
    # Note: Ensure these images are preprocessed from the original DZT files.
    source_path = "data/dataset/images/test"

    # Perform inference on the GPR profiles    
    results = model.predict(
        source=source_path,
        conf=0.25,                   
        iou=0.45,                    
        device=0,                    
        save=True,                  
        save_txt=True,               
        project="runs/detect",       
        name="GPR_Predictions"     
    )
    

if __name__ == "__main__":
    main()