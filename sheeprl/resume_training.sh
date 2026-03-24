#!/bin/bash

# ==============================================================================
# F-Zero DreamerV3 Training Script - DUAL GPU
# ==============================================================================
# This script is configured to use two GPUs for accelerated training.

echo "=========================================="
echo "F-Zero DreamerV3 Training Script (DUAL GPU)"
echo "=========================================="
echo ""

# Ask user whether to resume or start fresh
echo "Do you want to:"
echo "1) Resume from latest checkpoint"
echo "2) Start fresh training"
echo ""
read -p "Enter your choice (1 or 2): " choice

case $choice in
    1)
        echo "Resuming from latest checkpoint..."
        
        # Find the most recent experiment folder
        LATEST_FOLDER=$(ls -1d logs/runs/dreamer_v3/F-Zero-go/*/ 2>/dev/null | sort -V | tail -1 | xargs basename)
        
        if [ -z "$LATEST_FOLDER" ]; then
            echo "No experiment folder found in logs/runs/dreamer_v3/F-Zero-go/"
            echo "Starting fresh training instead..."
            RESUME_MODE=false
        else
            # Find latest version directory within experiment folder
            VERSION_DIR=$(ls -1d logs/runs/dreamer_v3/F-Zero-go/"$LATEST_FOLDER"/version_*/ 2>/dev/null | sort -V | tail -1)
            VERSION_NAME=$(basename "$VERSION_DIR")
            
            # Find latest checkpoint in the latest version directory
            CHECKPOINT=$(ls -1 "$VERSION_DIR"/checkpoint/ckpt_*.ckpt 2>/dev/null | sort -V | tail -1)
            
            if [ -z "$CHECKPOINT" ]; then
                echo "No checkpoint found in $LATEST_FOLDER/$VERSION_NAME!"
                echo "Starting fresh training instead..."
                RESUME_MODE=false
            else
                echo "Found checkpoint: $(basename "$CHECKPOINT")"
                RESUME_MODE=true
                EXPERIMENT_FOLDER="logs/runs/dreamer_v3/F-Zero-go/$LATEST_FOLDER/$VERSION_NAME"
                # Extract logger name from folder for resuming in same directory
                LOGGER_NAME="$LATEST_FOLDER"
            fi
        fi
        ;;
    2)
        echo "Starting fresh training..."
        RESUME_MODE=false
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""

# Create evaluation runner with torch.load patch (No changes needed)
cat > _eval_run.py << 'EOF'
import torch
original_load = torch.load
torch.load = lambda *args,  **kwargs: original_load(*args,  weights_only=False, **kwargs)
try:
    import lightning.fabric.utilities.cloud_io as cloud_io
    def patched_load(path, map_location=None):
        return original_load(path, map_location=map_location,  weights_only=False)
    cloud_io._load = patched_load
except:
    pass
from sheeprl.cli import evaluation
evaluation()
EOF

# Create the training wrapper with torch.load patch (No changes needed)
cat > _run_wrapper.py << 'EOF'
import torch
original_load = torch.load
torch.load = lambda *args,  **kwargs: original_load(*args,  weights_only=False, **kwargs)
try:
    import lightning.fabric.utilities.cloud_io as cloud_io
    def patched_load(path, map_location=None):
        return original_load(path, map_location=map_location,  weights_only=False)
    cloud_io._load = patched_load
except:
    pass
from sheeprl.cli import run
run()
EOF

# Make scripts executable
chmod +x _eval_run.py

# ==============================================================================
# DUAL GPU CONFIGURATION
# ==============================================================================
# MODIFIED: Parameters for 2x 3090 training
GPU_COUNT=1
CUDA_VISIBLE="1"
MODEL="dreamer_v3_L"   # Using the L model
PER_RANK_BATCH_SIZE=32  # Batch size for single GPU
SEQ_LENGTH=128          # Sequence length.
LEARNING_STARTS=10000   # Increased to collect more data for the larger total batch size.

# Build training command
if [ "$RESUME_MODE" = true ]; then
    TRAINING_CMD="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE python _run_wrapper.py \
        exp=dreamer_v3_fzero \
        algo=$MODEL \
        fabric.devices=$GPU_COUNT \
        env.num_envs=1 \
        algo.per_rank_batch_size=$PER_RANK_BATCH_SIZE \
        algo.learning_starts=$LEARNING_STARTS \
        algo.per_rank_sequence_length=$SEQ_LENGTH \
        env.capture_video=false \
        metric.log_every=100 \
        checkpoint.resume_from=\"$CHECKPOINT\" \
        run_name=\"$LOGGER_NAME\" \
        metric.logger.root_dir=logs/runs/dreamer_v3/F-Zero-go"
else
    TRAINING_CMD="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE python _run_wrapper.py \
        exp=dreamer_v3_fzero \
        algo=$MODEL \
        fabric.devices=$GPU_COUNT \
        env.num_envs=1 \
        algo.per_rank_batch_size=$PER_RANK_BATCH_SIZE \
        algo.learning_starts=$LEARNING_STARTS \
        algo.per_rank_sequence_length=$SEQ_LENGTH \
        env.capture_video=false \
        metric.log_every=100"
fi

# Start training
echo "Starting training with the following parameters:"
echo "------------------------------------------------"
echo "GPUs: $GPU_COUNT"
echo "Model: $MODEL"
echo "Sequence Length: $SEQ_LENGTH"
echo "Batch Size per GPU: $PER_RANK_BATCH_SIZE"
echo "Total Batch Size: $(($PER_RANK_BATCH_SIZE * $GPU_COUNT))"
if [ "$RESUME_MODE" = true ]; then
    echo "Resuming in folder: $LOGGER_NAME"
fi
echo "------------------------------------------------"
eval $TRAINING_CMD
TRAINING_EXIT_CODE=$?

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training stopped."
fi

# Cleanup
rm -f _run_wrapper.py _eval_run.py

echo "All done!"
