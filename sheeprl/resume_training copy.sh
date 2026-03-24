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

# Create evaluation script that will be called after each checkpoint
# MODIFIED: Fixed video folder detection
cat > _eval_checkpoint.py << 'EOF'
#!/usr/bin/env python3
import sys
import os
import subprocess
import time
import re
from pathlib import Path
import random
import shutil

def parse_reward_from_output(output_text):
    """Extract the reward/return value from evaluation output."""
    # Look for patterns like "Reward: X.XX" or "Return: X.XX" or "reward: X.XX"
    patterns = [
        r'[Rr]eward[:\s]+(-?\d+\.?\d*)',
        r'[Rr]eturn[:\s]+(-?\d+\.?\d*)',
        r'[Ee]pisode[_\s]+[Rr]eward[:\s]+(-?\d+\.?\d*)',
        r'[Tt]est[_\s]+[Rr]eward[:\s]+(-?\d+\.?\d*)',
        r'[Ee]val[_\s]+[Rr]eward[:\s]+(-?\d+\.?\d*)',
        r'[Aa]verage[_\s]+[Rr]eward[:\s]+(-?\d+\.?\d*)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output_text)
        if match:
            return float(match.group(1))
    
    # If no pattern matches, return None
    return None

def move_and_rename_videos(video_folder, checkpoint_name, reward, trial_num):
    """Move videos to Videos folder and rename with checkpoint info."""
    if not video_folder.exists():
        return False
    
    # Create Videos directory if it doesn't exist
    videos_dir = Path("/root/SheepRL/sheeprl/Videos")
    videos_dir.mkdir(exist_ok=True)
    
    videos = list(video_folder.glob("*.mp4"))
    moved_count = 0
    
    for video_path in videos:
        # Get the original video name
        original_name = video_path.stem
        extension = video_path.suffix
        
        # Create new name with checkpoint, reward, and trial info
        if reward is not None:
            new_name = f"{checkpoint_name}_best_trial_{trial_num}_reward_{reward:.2f}{extension}"
        else:
            new_name = f"{checkpoint_name}_best_trial_{trial_num}{extension}"
        
        # Move and rename the file
        new_path = videos_dir / new_name
        try:
            shutil.move(str(video_path), str(new_path))
            print(f"  Moved video to: {new_path}")
            moved_count += 1
        except Exception as e:
            print(f"  Failed to move {video_path.name}: {e}")
    
    return moved_count > 0

def get_existing_eval_versions(log_dir):
    """Get a set of existing evaluation version folders."""
    eval_path = Path(log_dir) / "evaluation"
    if not eval_path.exists():
        return set()
    return set(p.name for p in eval_path.glob("version_*") if p.is_dir())

def find_new_eval_version(log_dir, existing_versions):
    """Find the new evaluation version folder that was created."""
    eval_path = Path(log_dir) / "evaluation"
    if not eval_path.exists():
        return None
    
    current_versions = set(p.name for p in eval_path.glob("version_*") if p.is_dir())
    new_versions = current_versions - existing_versions
    
    if new_versions:
        # Get the highest numbered new version
        version_nums = []
        for v in new_versions:
            try:
                num = int(v.split('_')[1])
                version_nums.append((num, v))
            except:
                pass
        if version_nums:
            version_nums.sort(reverse=True)
            return version_nums[0][1]
    
    return None

def run_single_trial(checkpoint_path, seed, capture_video=False):
    """Run a single evaluation trial with the given seed."""
    # Create evaluation command
    eval_cmd = [
        sys.executable, "_eval_run.py",
        f"checkpoint_path={checkpoint_path}",
        f"env.capture_video={str(capture_video).lower()}",
        "fabric.accelerator=gpu",
        "+env.frame_skip=4",
        "+env.capture_video_fps=60",
        "+fabric.devices=1",
        f"seed={seed}"
    ]
    
    # Set environment variables - Pin evaluation to a single GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    
    try:
        # Run evaluation and capture output
        result = subprocess.run(eval_cmd, cwd="/root/SheepRL/sheeprl", env=env, 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            # Parse reward from output
            reward = parse_reward_from_output(result.stdout + result.stderr)
            return reward, True
        else:
            return None, False
    except subprocess.TimeoutExpired:
        return None, False
    except Exception as e:
        return None, False

def run_evaluation(checkpoint_path, log_dir):
    """Run 20 evaluation trials and generate video for the best one."""
    print(f"\nEvaluating checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"Running 20 trials to find best episode...")
    
    checkpoint_name = os.path.basename(checkpoint_path).replace('.ckpt', '')
    
    # Run 20 trials without video to find the best one
    trials_results = []
    for i in range(20):
        seed = random.randint(0, 999999)
        print(f"  Trial {i+1}/20 (seed={seed})...", end='', flush=True)
        
        reward, success = run_single_trial(checkpoint_path, seed, capture_video=False)
        
        if success and reward is not None:
            trials_results.append((i+1, seed, reward))
            print(f" Reward: {reward:.2f}")
        else:
            print(" Failed")
    
    if not trials_results:
        print("✗ All trials failed!")
        return
    
    # Find the best trial
    best_trial = max(trials_results, key=lambda x: x[2])
    best_trial_num, best_seed, best_reward = best_trial
    
    print(f"\n✓ Best trial: #{best_trial_num} with reward {best_reward:.2f} (seed={best_seed})")
    print(f"Generating video for best trial...")
    
    # Get existing evaluation versions before video generation
    existing_versions = get_existing_eval_versions(log_dir)
    
    # Re-run the best trial with video capture
    _, success = run_single_trial(checkpoint_path, best_seed, capture_video=True)
    
    if success:
        print(f"✓ Video generation completed!")
        
        # Wait a bit for video file to be written
        time.sleep(2)
        
        # Find the new evaluation version folder
        new_version = find_new_eval_version(log_dir, existing_versions)
        
        if new_version:
            video_folder = Path(log_dir) / "evaluation" / new_version / "test_videos"
            print(f"Looking for videos in: {video_folder}")
            
            if video_folder.exists():
                # Move and rename videos to the Videos directory
                if move_and_rename_videos(video_folder, checkpoint_name, best_reward, best_trial_num):
                    print(f"✓ Video moved to /root/SheepRL/sheeprl/Videos/")
                else:
                    print(f"✗ Failed to move video files")
            else:
                print(f"✗ Video folder not found: {video_folder}")
                # Try to find any test_videos folder in the new version
                new_version_path = Path(log_dir) / "evaluation" / new_version
                test_video_folders = list(new_version_path.glob("**/test_videos"))
                if test_video_folders:
                    print(f"Found video folder at: {test_video_folders[0]}")
                    if move_and_rename_videos(test_video_folders[0], checkpoint_name, best_reward, best_trial_num):
                        print(f"✓ Video moved to /root/SheepRL/sheeprl/Videos/")
        else:
            print(f"✗ Could not find new evaluation version folder")
            # List all evaluation folders for debugging
            eval_path = Path(log_dir) / "evaluation"
            if eval_path.exists():
                versions = sorted(eval_path.glob("version_*"))
                print(f"Existing evaluation versions: {[v.name for v in versions[-5:]]}")
    else:
        print(f"✗ Failed to generate video for best trial")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: _eval_checkpoint.py <checkpoint_path> <log_dir>")
        sys.exit(1)
    
    run_evaluation(sys.argv[1], sys.argv[2])
EOF

# Create checkpoint monitor script (No changes needed)
cat > _monitor_checkpoints.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
import time
import subprocess
from pathlib import Path

def monitor_checkpoints(checkpoint_dir, log_dir):
    """Monitor checkpoint directory and run evaluation when new checkpoints appear."""
    print(f"Monitoring checkpoints in: {checkpoint_dir}")
    processed = set()
    if os.path.exists(checkpoint_dir):
        for ckpt in Path(checkpoint_dir).glob("ckpt_*.ckpt"):
            processed.add(ckpt.name)
    while True:
        try:
            if os.path.exists(checkpoint_dir):
                current_checkpoints = set(ckpt.name for ckpt in Path(checkpoint_dir).glob("ckpt_*.ckpt"))
                new_checkpoints = current_checkpoints - processed
                for ckpt_name in sorted(new_checkpoints):
                    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
                    print(f"\n🎬 New checkpoint detected: {ckpt_name}")
                    subprocess.run([sys.executable, "_eval_checkpoint.py", ckpt_path, log_dir])
                    processed.add(ckpt_name)
            time.sleep(30)
        except KeyboardInterrupt:
            print("\nStopping checkpoint monitor...")
            break
        except Exception as e:
            print(f"Monitor error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: _monitor_checkpoints.py <checkpoint_dir> <log_dir>")
        sys.exit(1)
    monitor_checkpoints(sys.argv[1], sys.argv[2])
EOF

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
chmod +x _eval_checkpoint.py _monitor_checkpoints.py

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

# Start training in background and capture its PID
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
eval $TRAINING_CMD &
TRAINING_PID=$!

# Wait a bit for training to start and create directories
echo "Waiting for training to initialize..."
sleep 15

# Find the experiment folder (for new or resumed training)
# Wait for training to create the new version directory
echo "Waiting for training to create new version directory..."
MAX_WAIT=60
WAIT_COUNT=0

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if [ "$RESUME_MODE" = true ]; then
        # For resumed training, find the latest version in the same experiment folder
        VERSION_DIR=$(ls -1d logs/runs/dreamer_v3/F-Zero-go/"$LOGGER_NAME"/version_*/ 2>/dev/null | sort -V | tail -1)
    else
        # For fresh training, find the newest folder and latest version
        LATEST_FOLDER=$(ls -1t logs/runs/dreamer_v3/F-Zero-go/ | head -1)
        if [ -n "$LATEST_FOLDER" ]; then
            VERSION_DIR=$(ls -1d logs/runs/dreamer_v3/F-Zero-go/"$LATEST_FOLDER"/version_*/ 2>/dev/null | sort -V | tail -1)
        fi
    fi
    
    if [ -n "$VERSION_DIR" ] && [ -d "$VERSION_DIR" ]; then
        EXPERIMENT_FOLDER="$VERSION_DIR"
        echo "Found new version directory: $EXPERIMENT_FOLDER"
        break
    fi
    
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
    echo "Still waiting... ($WAIT_COUNT/$MAX_WAIT seconds)"
done

if [ -n "$EXPERIMENT_FOLDER" ]; then
    CHECKPOINT_DIR="$EXPERIMENT_FOLDER/checkpoint"
    echo "Monitoring checkpoints in: $CHECKPOINT_DIR"
    
    # Start checkpoint monitor in background
    python _monitor_checkpoints.py "$CHECKPOINT_DIR" "$EXPERIMENT_FOLDER" &
    MONITOR_PID=$!
    
    echo ""
    echo "=========================================="
    echo "Training started!"
    echo "- Training PID: $TRAINING_PID"
    echo "- Monitor PID: $MONITOR_PID"
    echo "- Evaluation: 20 trials per checkpoint"
    echo "- Videos: Only best trial saved to /root/SheepRL/sheeprl/Videos/"
    echo "- Press Ctrl+C to stop both training and monitoring"
    echo "=========================================="
    echo ""
    
    # Wait for training to complete or user interrupt
    wait $TRAINING_PID
    TRAINING_EXIT_CODE=$?
    
    # Stop the monitor
    kill $MONITOR_PID 2>/dev/null
    
    if [ $TRAINING_EXIT_CODE -eq 0 ]; then
        echo "Training completed successfully!"
    else
        echo "Training stopped."
    fi
else
    echo "Warning: Could not determine experiment folder for monitoring"
    echo "Training is running but checkpoint monitoring is disabled"
    wait $TRAINING_PID
fi

# Cleanup
rm -f _run_wrapper.py _eval_run.py _eval_checkpoint.py _monitor_checkpoints.py

echo "All done!"