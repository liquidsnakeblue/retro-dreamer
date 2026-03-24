#!/bin/bash

# Script to generate evaluation videos for ALL checkpoints in F-Zero runs

echo "=========================================="
echo "F-Zero Video Generation Script"
echo "=========================================="

# Base directory for all F-Zero runs
BASE_DIR="/root/SheepRL/sheeprl/logs/runs/dreamer_v3/F-Zero-go"
VIDEO_DIR="$BASE_DIR/Videos"

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$VIDEO_DIR/run_$TIMESTAMP"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Function to extract checkpoint step from path
get_checkpoint_step() {
    local checkpoint_path=$1
    echo "$checkpoint_path" | sed -n 's/.*ckpt_\([0-9]*\)_[0-9]*.ckpt/\1/p'
}

# Create evaluation script
cat > _eval_checkpoint.py << 'EOF'
#!/usr/bin/env python3
import sys
import os
import subprocess
import time
from pathlib import Path

def run_evaluation(checkpoint_path, output_video_path):
    """Run evaluation and generate video for a specific checkpoint."""
    print(f"\nGenerating video for: {os.path.basename(checkpoint_path)}")
    
    # Get the parent directory where evaluation folders are created
    parent_dir = Path(checkpoint_path).parent.parent
    eval_dir = parent_dir / "evaluation"
    
    # Get list of existing evaluation versions before running
    existing_versions = set()
    if eval_dir.exists():
        existing_versions = set(p.name for p in eval_dir.glob("version_*"))
    
    # Create evaluation command
    eval_cmd = [
        sys.executable, "_eval_run.py",
        f"checkpoint_path={checkpoint_path}",
        "env.capture_video=True",
        "fabric.accelerator=gpu",
        "+fabric.devices=1",
        "seed=42"
    ]
    
    # Set environment variables
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0 for evaluation
    
    try:
        # Run evaluation
        result = subprocess.run(eval_cmd, cwd="/root/SheepRL/sheeprl", env=env, 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✓ Evaluation completed successfully!")
            
            # Find the NEW evaluation folder that was just created
            time.sleep(1)  # Brief pause to ensure filesystem is updated
            
            if eval_dir.exists():
                new_versions = set(p.name for p in eval_dir.glob("version_*")) - existing_versions
                
                if new_versions:
                    # Get the newly created version folder
                    new_version = sorted(new_versions)[-1]
                    video_path = eval_dir / new_version / "test_videos"
                    
                    # Find the video in the new folder
                    videos = list(video_path.glob("*.mp4"))
                    if videos:
                        # Copy the first (and usually only) video
                        import shutil
                        shutil.copy2(videos[0], output_video_path)
                        print(f"✓ Video saved to: {output_video_path}")
                        return True
                    else:
                        print(f"✗ No video found in new evaluation folder: {video_path}")
                        return False
                else:
                    print(f"✗ No new evaluation folder was created")
                    return False
            else:
                print(f"✗ Evaluation directory does not exist")
                return False
        else:
            print(f"✗ Evaluation failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ Evaluation timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: _eval_checkpoint.py <checkpoint_path> <output_video_path>")
        sys.exit(1)
    
    success = run_evaluation(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
EOF

# Create evaluation runner with torch.load patch
cat > _eval_run.py << 'EOF'
import torch
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, weights_only=False, **kwargs)
try:
    import lightning.fabric.utilities.cloud_io as cloud_io
    def patched_load(path, map_location=None):
        return original_load(path, map_location=map_location, weights_only=False)
    cloud_io._load = patched_load
except:
    pass
from sheeprl.cli import evaluation
evaluation()
EOF

# Make scripts executable
chmod +x _eval_checkpoint.py

# Find all checkpoints
echo "Finding all checkpoints..."
echo ""

checkpoint_count=0

# Find all checkpoint files
find "$BASE_DIR" -name "ckpt_*_0.ckpt" -type f | sort | while read checkpoint_path; do
    checkpoint_count=$((checkpoint_count + 1))
    
    # Extract info from path
    run_dir=$(echo "$checkpoint_path" | grep -oE '2025-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}_dreamer_v3_F-Zero-go_[0-9]+')
    version=$(echo "$checkpoint_path" | grep -oE 'version_[0-9]+' | head -1 | grep -oE '[0-9]+')
    checkpoint_step=$(get_checkpoint_step "$checkpoint_path")
    
    # Create output filename
    output_filename="${run_dir}_v${version}_step${checkpoint_step}.mp4"
    output_path="$OUTPUT_DIR/$output_filename"
    
    echo "[$checkpoint_count] Processing checkpoint:"
    echo "    Run: $run_dir"
    echo "    Version: $version"
    echo "    Step: $checkpoint_step"
    echo "    Output: $output_filename"
    
    # Generate video
    python _eval_checkpoint.py "$checkpoint_path" "$output_path"
    
    echo "    ---"
    
    # Brief pause between evaluations
    sleep 2
done

# Clean up temporary scripts
rm -f _eval_checkpoint.py _eval_run.py

echo ""
echo "=========================================="
echo "Video Generation Complete!"
echo "=========================================="
echo "Videos saved in: $OUTPUT_DIR"
echo ""
ls -la "$OUTPUT_DIR"/*.mp4 2>/dev/null | wc -l | xargs echo "Total videos generated:"