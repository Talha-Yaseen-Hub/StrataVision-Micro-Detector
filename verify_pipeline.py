import os
import sys
import shutil
import yaml
from pathlib import Path
from PIL import Image, ImageDraw

# Add 'src' directory to python path so we can import internal modules
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

def print_header(title):
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    print("=" * 80)

def verify_imports():
    print_header("Step 1: Verifying Core Dependencies")
    dependencies = [
        ("torch", "PyTorch"),
        ("yaml", "PyYAML"),
        ("ultralytics", "Ultralytics YOLO"),
        ("sahi", "SAHI (Slicing Aided Hyper Inference)"),
        ("pycocotools", "COCO Evaluation Tools"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("matplotlib", "Matplotlib")
    ]
    
    missing = []
    for module_name, desc in dependencies:
        try:
            __import__(module_name)
            print(f"  [✓] {desc} ({module_name}) is successfully installed.")
        except ImportError as e:
            print(f"  [✗] {desc} ({module_name}) is MISSING. Error: {e}")
            missing.append(module_name)
            
    if missing:
        print("\n[ERROR] Some dependencies are missing. Please run:")
        print("        pip install -r requirements.txt")
        sys.exit(1)
    
    import torch
    print(f"\n  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Device   : {torch.cuda.get_device_name(0)}")
    print("  All core imports validated successfully!")

def create_synthetic_dataset():
    print_header("Step 2: Creating Synthetic Toy Dataset")
    
    toy_dir = ROOT_DIR / "dataset_toy"
    if toy_dir.exists():
        print(f"  Removing existing toy dataset directory: {toy_dir}")
        shutil.rmtree(toy_dir)
        
    # Create directory structure
    for split in ["train", "val", "test"]:
        (toy_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (toy_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    (toy_dir / "val" / "annotations").mkdir(parents=True, exist_ok=True)
    
    # Let's generate a few synthetic images
    # Class definitions: 0: drone, 1: bird, 2: vehicle, 3: person, 4: small_component
    # We will draw simple color shapes:
    # Drone (red square), Bird (blue circle), Vehicle (green rectangle)
    
    def generate_image_and_labels(img_path, label_path, objects):
        # Create solid gray background
        img = Image.new("RGB", (640, 640), color=(128, 128, 128))
        draw = ImageDraw.Draw(img)
        
        yolo_labels = []
        for class_id, x1, y1, x2, y2 in objects:
            # Draw shapes
            if class_id == 0: # Drone (red square)
                draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0), outline=(0, 0, 0))
            elif class_id == 1: # Bird (blue circle/ellipse)
                draw.ellipse([x1, y1, x2, y2], fill=(0, 0, 255), outline=(0, 0, 0))
            else: # Vehicle (green rectangle)
                draw.rectangle([x1, y1, x2, y2], fill=(0, 255, 0), outline=(0, 0, 0))
                
            # Compute YOLO coords (normalized center x, center y, width, height)
            w = (x2 - x1)
            h = (y2 - y1)
            cx = x1 + w / 2
            cy = y1 + h / 2
            
            yolo_labels.append(f"{class_id} {cx/640:.6f} {cy/640:.6f} {w/640:.6f} {h/640:.6f}")
            
        img.save(img_path)
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_labels))

    # Train objects
    train_objects_0 = [(0, 100, 100, 120, 120), (1, 300, 200, 316, 216)] # Drone & Bird
    train_objects_1 = [(2, 400, 400, 460, 430)] # Vehicle
    train_objects_2 = [(0, 50, 500, 70, 520), (1, 500, 100, 512, 112)] # Drone & Bird (tiny)
    
    # Val objects
    val_objects_0 = [(0, 120, 120, 140, 140), (2, 200, 450, 260, 480)] # Drone & Vehicle
    val_objects_1 = [(1, 310, 210, 325, 225)] # Bird
    
    # Generate files
    generate_image_and_labels(toy_dir / "train/images/train_0.jpg", toy_dir / "train/labels/train_0.txt", train_objects_0)
    generate_image_and_labels(toy_dir / "train/images/train_1.jpg", toy_dir / "train/labels/train_1.txt", train_objects_1)
    generate_image_and_labels(toy_dir / "train/images/train_2.jpg", toy_dir / "train/labels/train_2.txt", train_objects_2)
    
    generate_image_and_labels(toy_dir / "val/images/val_0.jpg", toy_dir / "val/labels/val_0.txt", val_objects_0)
    generate_image_and_labels(toy_dir / "val/images/val_1.jpg", toy_dir / "val/labels/val_1.txt", val_objects_1)
    
    # Create config files
    toy_dataset_config = {
        "path": str(toy_dir),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 5,
        "names": {
            0: "drone",
            1: "bird",
            2: "vehicle",
            3: "person",
            4: "small_component"
        }
    }
    
    toy_config_path = ROOT_DIR / "config" / "dataset_toy.yaml"
    with open(toy_config_path, "w") as f:
        yaml.safe_dump(toy_dataset_config, f)
        
    print(f"  Created synthetic dataset in: {toy_dir}")
    print(f"  Created dataset config in: {toy_config_path}")

def run_dataset_validation():
    print_header("Step 3: Running Dataset Validator")
    from src.utils.dataset_utils import validate_dataset
    validate_dataset(dataset_dir="dataset_toy", dataset_yaml="config/dataset_toy.yaml")
    print("  Dataset validator verified successfully!")

def run_coco_conversion():
    print_header("Step 4: Running YOLO-to-COCO Conversion")
    from src.utils.coco_converter import yolo_to_coco
    coco_json = yolo_to_coco(
        images_dir="dataset_toy/val/images",
        labels_dir="dataset_toy/val/labels",
        output_json="dataset_toy/val/annotations/instances_val.json",
        dataset_yaml="config/dataset_toy.yaml"
    )
    print(f"  COCO JSON generated: {coco_json}")
    if coco_json and Path(coco_json).exists():
        print("  COCO converter verified successfully!")
    else:
        print("[ERROR] COCO JSON conversion failed!")
        sys.exit(1)

def find_trained_weights():
    candidates = [
        ROOT_DIR / "runs" / "detect" / "detection_models_toy" / "verification_run" / "weights" / "best.pt",
        ROOT_DIR / "runs" / "detect" / "detection_models_toy" / "verification_run" / "weights" / "last.pt",
        ROOT_DIR / "detection_models_toy" / "verification_run" / "weights" / "best.pt",
        ROOT_DIR / "detection_models_toy" / "verification_run" / "weights" / "last.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not locate trained weights in expected directories.")

def run_training_smoke_test():
    print_header("Step 5: Running Model Setup & Training Smoke Test")
    from src.train import train_small_target_detector
    
    # Run a 1-epoch training run on CPU to make sure the network builds and trains
    train_small_target_detector(
        model_config="models/yolov8n-p2.yaml",
        data_config="config/dataset_toy.yaml",
        epochs=1,
        imgsz=640,
        batch=2,
        device="cpu",
        project="detection_models_toy",
        name="verification_run"
    )
    print("  Training pipeline and architecture model initialization verified successfully!")

def run_inference_smoke_test():
    print_header("Step 6: Running Standard Inference Script")
    from src.inference import run_inference
    
    weights_path = find_trained_weights()
    print(f"  Using model weights at: {weights_path}")
        
    run_inference(
        source="dataset_toy/val/images",
        model_path=str(weights_path),
        imgsz=640,
        device="cpu",
        project="output_toy",
        name="standard_inference"
    )
    print("  Standard inference verified successfully!")

def run_sahi_inference_smoke_test():
    print_header("Step 7: Running SAHI Sliced Inference Script")
    from src.sahi_inference import run_sahi_inference
    
    weights_path = find_trained_weights()
        
    run_sahi_inference(
        image_path="dataset_toy/val/images/val_0.jpg",
        model_weights=str(weights_path),
        device="cpu",
        slice_height=320,
        slice_width=320,
        export_dir="output_toy",
        perform_standard_prediction=False
    )
    print("  SAHI sliced inference verified successfully!")

def run_evaluation_smoke_test():
    print_header("Step 8: Running SAHI COCO Evaluation Metrics Engine")
    from src.evaluate import run_sahi_validation, run_coco_evaluation
    
    weights_path = find_trained_weights()
        
    pred_json = run_sahi_validation(
        val_images_dir="dataset_toy/val/images",
        coco_ground_truth_json="dataset_toy/val/annotations/instances_val.json",
        model_weights=str(weights_path),
        device="cpu",
        slice_height=320,
        slice_width=320,
        project="metrics_toy",
        name="evaluation_run"
    )
    
    run_coco_evaluation(
        coco_ground_truth_json="dataset_toy/val/annotations/instances_val.json",
        prediction_results_json=pred_json,
        output_dir="metrics_toy"
    )
    print("  Evaluation engine verified successfully!")

def cleanup():
    print_header("Step 9: Cleaning Up Toy Files (Optional)")
    
    ans = input("Do you want to clean up the generated toy files? (y/n): ").strip().lower()
    if ans == "y" or ans == "yes":
        dirs_to_clean = [
            "dataset_toy",
            "detection_models_toy",
            "output_toy",
            "metrics_toy",
            "runs/detect/detection_models_toy",
            "runs/sahi"
        ]
        files_to_clean = ["config/dataset_toy.yaml"]
        
        for d in dirs_to_clean:
            path = ROOT_DIR / d
            if path.exists():
                print(f"  Removing directory: {path}")
                shutil.rmtree(path)
                
        for f in files_to_clean:
            path = ROOT_DIR / f
            if path.exists():
                print(f"  Removing file: {path}")
                path.unlink()
                
        print("  Cleanup completed successfully!")
    else:
        print("  Skipping cleanup. You can manually inspect the files in:")
        print("    - dataset_toy/         (Toy images and annotations)")
        print("    - config/dataset_toy.yaml (Toy dataset configuration)")
        print("    - detection_models_toy/(Trained weights and plots)")
        print("    - output_toy/          (Inference visual output)")
        print("    - metrics_toy/         (COCO evaluation output)")

def main():
    print("=" * 80)
    print("          ADVANCED SMALL OBJECT DETECTION PIPELINE VERIFICATION")
    print("=" * 80)
    
    verify_imports()
    create_synthetic_dataset()
    run_dataset_validation()
    run_coco_conversion()
    run_training_smoke_test()
    run_inference_smoke_test()
    run_sahi_inference_smoke_test()
    run_evaluation_smoke_test()
    cleanup()
    
    print("\n" + "=" * 80)
    print("                  VERIFICATION COMPLETE - PIPELINE IS WORKING! 🎉")
    print("=" * 80)

if __name__ == "__main__":
    main()
