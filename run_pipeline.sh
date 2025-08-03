#!/bin/bash

# =============================================================================
# Machine Learning Pipeline Script
# =============================================================================
# This script runs the complete ML pipeline: Train -> Test -> Visualize
# Modify the variables below to customize your experiment
# =============================================================================

# =============================================================================
# CONFIGURATION VARIABLES - MODIFY THESE AS NEEDED
# =============================================================================

# Dataset Configuration
DATASET_NAME="gear"                           # Name of your dataset
DATA_ROOT="datasets/Gear"                    # Path to dataset root directory
# For a new dataset with same structure, change these:
# DATASET_NAME="new_gear_dataset"
# DATA_ROOT="datasets/NewGearDataset"

# Training Parameters
EPOCHS=50                                     # Number of training epochs
BATCH_SIZE=8                                 # Batch size for training
LEARNING_RATE=0.001                          # Learning rate
WEIGHT_DECAY=0.0001                          # Weight decay
OPTIMIZER="adam"                              # Optimizer: adam, adamw, sgd
IMAGE_SIZE=512                                # Input image size (square)

# Model Configuration
MODEL="seg_unet"                              # Model: unet, seg_unet
BILINEAR_FLAG=""                              # Add "--bilinear" to enable bilinear upsampling
DROPOUT=0.1                                   # Dropout rate

# Loss Configuration
CE_WEIGHT=1.0                                 # Cross entropy loss weight
DICE_WEIGHT=1.0                               # Dice loss weight
FOCAL_WEIGHT=0.0                              # Focal loss weight
CLASS_WEIGHTS=""                              # e.g., "1.0,2.0,1.5" or leave empty

# Hardware Configuration
NUM_WORKERS=4                                 # Number of data loading workers
DEVICE="auto"                                 # Device: auto, cpu, cuda
SEED=42                                       # Random seed

# Output Configuration
BASE_OUTPUT_DIR="outputs"                     # Base directory for all outputs
SAVE_FREQ=10                                  # Save checkpoint every N epochs
VAL_FREQ=5                                    # Validate every N epochs

# Test Configuration
TEST_SPLIT="test"                             # Split to test on: test, val
SAVE_PREDICTIONS="--save_predictions"         # Add this flag to save prediction visualizations
SAVE_CONFUSION_MATRIX="--save_confusion_matrix"  # Add this flag to save confusion matrix

# Visualization Configuration
NUM_SAMPLES=10                                # Number of samples to visualize
VIS_SPLIT="test"                              # Split to visualize: test, val, train
SAVE_INDIVIDUAL="--save_individual"           # Save individual predictions
SAVE_GRID="--save_grid"                       # Save prediction grid
SHOW_CONFIDENCE="--show_confidence"           # Show confidence maps

# Debug Mode (set to true for faster testing with limited data)
DEBUG_MODE=false                              # Set to true to enable debug mode
DEBUG_SAMPLES=20                              # Number of samples in debug mode

# =============================================================================
# DERIVED VARIABLES - DO NOT MODIFY
# =============================================================================

# Create experiment name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="${DATASET_NAME}_${MODEL}_${TIMESTAMP}"
EXPERIMENT_DIR="${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}"

# Set class weights argument if provided
if [ -n "$CLASS_WEIGHTS" ]; then
    CLASS_WEIGHTS_ARG="--class_weights $CLASS_WEIGHTS"
else
    CLASS_WEIGHTS_ARG=""
fi

# Set debug arguments if enabled
if [ "$DEBUG_MODE" = true ]; then
    DEBUG_ARGS="--debug --debug_samples $DEBUG_SAMPLES"
    echo "🐛 DEBUG MODE ENABLED - Using limited data for faster testing"
else
    DEBUG_ARGS=""
fi

# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

echo "🚀 Starting ML Pipeline for $EXPERIMENT_NAME"
echo "============================================="
echo "📁 Dataset: $DATASET_NAME ($DATA_ROOT)"
echo "🤖 Model: $MODEL"
echo "📊 Epochs: $EPOCHS, Batch Size: $BATCH_SIZE"
echo "💾 Output Directory: $EXPERIMENT_DIR"
echo "============================================="

# Create experiment directory
mkdir -p "$EXPERIMENT_DIR"

# Log the configuration
cat > "$EXPERIMENT_DIR/pipeline_config.txt" << EOF
Pipeline Configuration - $TIMESTAMP
=====================================
Dataset: $DATASET_NAME
Data Root: $DATA_ROOT
Model: $MODEL
Epochs: $EPOCHS
Batch Size: $BATCH_SIZE
Learning Rate: $LEARNING_RATE
Image Size: $IMAGE_SIZE
Debug Mode: $DEBUG_MODE
=====================================
EOF

# =============================================================================
# STEP 1: TRAINING
# =============================================================================

echo ""
echo "🏋️ STEP 1: TRAINING"
echo "==================="

TRAIN_CMD="python train.py \
    --data_root '$DATA_ROOT' \
    --save_dir '$BASE_OUTPUT_DIR' \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --optimizer $OPTIMIZER \
    --model $MODEL \
    --image_size $IMAGE_SIZE \
    --dropout $DROPOUT \
    --ce_weight $CE_WEIGHT \
    --dice_weight $DICE_WEIGHT \
    --focal_weight $FOCAL_WEIGHT \
    --num_workers $NUM_WORKERS \
    --device $DEVICE \
    --seed $SEED \
    --save_freq $SAVE_FREQ \
    --val_freq $VAL_FREQ \
    $BILINEAR_FLAG \
    $CLASS_WEIGHTS_ARG \
    $DEBUG_ARGS"

echo "Executing: $TRAIN_CMD"
echo ""

if eval $TRAIN_CMD; then
    echo "✅ Training completed successfully!"
    
    # Find the most recent experiment directory (latest training run)
    LATEST_EXPERIMENT=$(find "$BASE_OUTPUT_DIR" -name "${DATASET_NAME}_${MODEL}_*" -type d | sort | tail -1)
    BEST_MODEL_PATH="$LATEST_EXPERIMENT/checkpoints/best_model.pth"
    
    if [ -f "$BEST_MODEL_PATH" ]; then
        echo "📍 Best model found at: $BEST_MODEL_PATH"
    else
        echo "⚠️ Warning: Best model not found. Using latest checkpoint."
        # Find the latest checkpoint
        BEST_MODEL_PATH=$(find "$LATEST_EXPERIMENT/checkpoints" -name "checkpoint_epoch_*.pth" | sort -V | tail -1)
        if [ -f "$BEST_MODEL_PATH" ]; then
            echo "📍 Using latest checkpoint: $BEST_MODEL_PATH"
        else
            echo "❌ Error: No checkpoints found!"
            exit 1
        fi
    fi
else
    echo "❌ Training failed!"
    exit 1
fi

# =============================================================================
# STEP 2: TESTING
# =============================================================================

echo ""
echo "🧪 STEP 2: TESTING"
echo "=================="

TEST_SAVE_DIR="$LATEST_EXPERIMENT/test_results"

TEST_CMD="python test.py \
    --data_root '$DATA_ROOT' \
    --checkpoint '$BEST_MODEL_PATH' \
    --save_dir '$TEST_SAVE_DIR' \
    --model $MODEL \
    --image_size $IMAGE_SIZE \
    --batch_size $BATCH_SIZE \
    --split $TEST_SPLIT \
    --num_workers $NUM_WORKERS \
    --device $DEVICE \
    --dropout $DROPOUT \
    $BILINEAR_FLAG \
    $SAVE_PREDICTIONS \
    $SAVE_CONFUSION_MATRIX \
    $DEBUG_ARGS"

echo "Executing: $TEST_CMD"
echo ""

if eval $TEST_CMD; then
    echo "✅ Testing completed successfully!"
else
    echo "❌ Testing failed!"
    exit 1
fi

# =============================================================================
# STEP 3: VISUALIZATION
# =============================================================================

echo ""
echo "🎨 STEP 3: VISUALIZATION"
echo "======================="

VIS_SAVE_DIR="$LATEST_EXPERIMENT/visualizations"

VIS_CMD="python visualize.py \
    --data_root '$DATA_ROOT' \
    --checkpoint '$BEST_MODEL_PATH' \
    --save_dir '$VIS_SAVE_DIR' \
    --model $MODEL \
    --image_size $IMAGE_SIZE \
    --batch_size $BATCH_SIZE \
    --split $VIS_SPLIT \
    --num_samples $NUM_SAMPLES \
    --num_workers $NUM_WORKERS \
    --device $DEVICE \
    --dropout $DROPOUT \
    $BILINEAR_FLAG \
    $SAVE_INDIVIDUAL \
    $SAVE_GRID \
    $SHOW_CONFIDENCE"

echo "Executing: $VIS_CMD"
echo ""

if eval $VIS_CMD; then
    echo "✅ Visualization completed successfully!"
else
    echo "❌ Visualization failed!"
    exit 1
fi

# =============================================================================
# PIPELINE COMPLETION
# =============================================================================

echo ""
echo "🎉 PIPELINE COMPLETED SUCCESSFULLY!"
echo "==================================="
echo "📁 All results saved to: $LATEST_EXPERIMENT"
echo ""
echo "📊 Result Summary:"
echo "  - Training logs: $LATEST_EXPERIMENT/logs/"
echo "  - Model checkpoints: $LATEST_EXPERIMENT/checkpoints/"
echo "  - Training results: $LATEST_EXPERIMENT/results/"
echo "  - Test results: $LATEST_EXPERIMENT/test_results/"
echo "  - Visualizations: $LATEST_EXPERIMENT/visualizations/"
echo ""
echo "🚀 To run with a different dataset:"
echo "  1. Edit DATASET_NAME and DATA_ROOT variables in this script"
echo "  2. Ensure your dataset has the same structure as the Gear dataset"
echo "  3. Run: ./run_pipeline.sh"
echo ""
echo "Happy Training! 🤖✨"