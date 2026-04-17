from ultralytics import YOLO

# Load the model
# Full list of available pre-trained models: https://docs.ultralytics.com/models/yolov8/#performance-metrics
model = YOLO("yolov8m.pt")  # Load the pre-trained model weights

# ─────────────────────────────────────────────────────────────────────────────
# model.train() — Full hyperparameter reference
# ─────────────────────────────────────────────────────────────────────────────
results = model.train(

    # ── Dataset ──────────────────────────────────────────────────────────────
    data='IA-8230365-8230196-7/data.yaml',  # Path to the dataset config file (YAML).
                                  # Must contain train/val/test paths and class names.

    # ── Core training settings ────────────────────────────────────────────────
    epochs=150,         # Total number of training epochs.
                        # More epochs = more learning, but risk of overfitting.
                        # Typical range: 50–300.

    patience=50,        # Early stopping: halt training if no improvement is seen
                        # for this many epochs. Set to 0 to disable.

    batch=16,           # Number of images per training batch.
                        # Higher = faster but needs more VRAM. Use -1 for AutoBatch.

    imgsz=640,          # Input image size (pixels). Images are resized to this square.
                        # Common values: 416, 512, 640, 1280.

    # ── Hardware ──────────────────────────────────────────────────────────────
    device="cpu",       # Device to train on.
                        # "cpu"       → CPU only (slow)
                        # 0           → first CUDA GPU
                        # [0, 1]      → multi-GPU (CUDA)
                        # "mps"       → Apple Silicon GPU

    workers=8,          # Number of DataLoader worker threads for loading images.
                        # Reduce if you see memory errors or CPU bottlenecks.

    # ── Checkpointing & output ────────────────────────────────────────────────
    project="runs/train",   # Root folder where training results are saved.
    name="foe-bot-exp1",    # Sub-folder name for this specific run.
                            # Results go to <project>/<name>/.

    exist_ok=False,     # If False, a new numbered folder is created when the name
                        # already exists. If True, the existing folder is overwritten.

    save=True,          # Save checkpoints (best.pt and last.pt) during training.
    save_period=-1,     # Save a checkpoint every N epochs. -1 = disabled.
                        # Example: save_period=10 saves every 10 epochs.

    # ── Transfer learning ─────────────────────────────────────────────────────
    pretrained=True,    # Start from pre-trained weights (recommended).
                        # Set to False to train from scratch.

    freeze=0,           # Freeze the first N layers of the backbone (do not update
                        # their weights). Useful for fine-tuning.
                        # Example: freeze=10 freezes the first 10 layers.

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer="auto",   # Optimiser algorithm. Options:
                        # "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"
                        # "auto" selects SGD or AdamW based on the model.

    lr0=0.01,           # Initial learning rate.
                        # Lower values (e.g. 0.001) are safer for fine-tuning.

    lrf=0.01,           # Final learning rate as a fraction of lr0.
                        # The scheduler decays lr0 → lr0 * lrf over training.

    momentum=0.937,     # SGD momentum / Adam beta1.
                        # Controls how much past gradients influence the update.

    weight_decay=0.0005, # L2 regularisation penalty — discourages large weights
                         # and helps prevent overfitting.

    warmup_epochs=3.0,  # Number of epochs for learning-rate warm-up at the start.
                        # LR gradually rises from 0 to lr0 during this phase.

    warmup_momentum=0.8, # Initial momentum during the warm-up phase.
    warmup_bias_lr=0.1,  # Initial learning rate for bias parameters during warm-up.

    # ── Loss weights ──────────────────────────────────────────────────────────
    box=7.5,            # Weight for the bounding-box regression loss.
                        # Increase to make the model focus more on box accuracy.

    cls=0.5,            # Weight for the classification loss.
    dfl=1.5,            # Weight for the Distribution Focal Loss (box refinement).

    # ── Augmentation(tiramos pois já temos augmentations no dataset) ──────────────────────────────────────────────────────────
    #hsv_h=0.015,        # Random hue shift (fraction of 360°). Adds colour variety.
    #hsv_s=0.7,          # Random saturation shift. Range: 0.0–1.0.
    #hsv_v=0.4,          # Random brightness (value) shift. Range: 0.0–1.0.

    #degrees=0.0,        # Random rotation range in degrees (e.g. 10 → ±10°).
    #translate=0.1,      # Random translation as a fraction of image size (e.g. 0.1 = 10%).
    #scale=0.5,          # Random scale factor (e.g. 0.5 means zoom between 50%–150%).
    #shear=0.0,          # Random shear angle in degrees.
    #perspective=0.0,    # Random perspective distortion. Range: 0.0–0.001.
    #flipud=0.0,         # Probability of vertical flip. 0.0 = disabled.
    #fliplr=0.5,         # Probability of horizontal flip. 0.5 = 50% chance per image.

    #mosaic=1.0,         # Probability of Mosaic augmentation (combines 4 images).
                        # Very effective for small object detection. 0.0 = disabled.

    #mixup=0.0,          # Probability of MixUp augmentation (blends 2 images).
                        # Useful for improving generalisation.

    #copy_paste=0.0,     # Probability of Copy-Paste augmentation (pastes objects
                        # from one image onto another). Good for instance segmentation.

    # ── Evaluation ────────────────────────────────────────────────────────────
    val=True,           # Run validation after each epoch to track mAP, loss, etc.

    plots=True,         # Generate and save training plots (loss curves, confusion
                        # matrix, PR curve) inside the results folder.

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed=0,             # Random seed for reproducibility. Use the same seed to get
                        # identical results across runs.

    deterministic=True, # Force deterministic CUDA operations.
                        # Ensures reproducibility but may slow down training slightly.

    # ── Misc ──────────────────────────────────────────────────────────────────
    resume=False,       # Resume training from the last saved checkpoint (last.pt).
                        # Set to the checkpoint path string to resume a specific run.

    amp=False,           # Automatic Mixed Precision (FP16). Speeds up training and
                        # reduces VRAM usage on supported hardware.

    fraction=1.0,       # Fraction of the training dataset to use.
                        # Example: 0.1 uses only 10% of images (useful for quick tests).

    profile=False,      # Profile ONNX and TensorRT speeds during training.
                        # Useful for benchmarking deployment targets.

    verbose=True,       # Print detailed training logs to the console.
)