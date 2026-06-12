from ultralytics import YOLO

def main():
        
    model = YOLO("models/yolo11-GRL-DPCT.yaml")
        
    results = model.train(
        data="data/data.yaml",       
        epochs=200,                  
        batch=16,                    
        device=0,                    
        optimizer="SGD",             
        lr0=0.001,                   
        weight_decay=0.0005,         
        momentum=0.937,              
        project="runs/train",        
        name="YOLO-GTGP-Train",      
    )
    print("Training Completed Successfully.")

if __name__ == "__main__":
    main()