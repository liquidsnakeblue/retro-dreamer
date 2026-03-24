# Quadratic Speed Reward System Implementation Guide

## Overview
This guide explains how to implement and use the enhanced quadratic speed reward system for F-Zero DreamerV3 agents. The new system replaces the binary reward (+0.1 if speed > 1.0) with a non-linear reward that scales quadratically with speed, providing better incentives for high-speed performance.

## Key Benefits
- **Non-linear scaling**: Rewards increase quadratically with speed
- **Better agent behavior**: Encourages agents to push for maximum speeds
- **Flexible configuration**: Multiple reward modes (binary, linear, quadratic, exponential)
- **Backward compatibility**: Existing configurations continue to work unchanged

## Configuration Examples

### 1. Quadratic Speed Reward (Recommended)
```json
{
  "reward": {
    "variables": {
      "speed": {
        "mode": "quadratic",
        "base_reward": 0.1,
        "max_speed": 500.0,
        "scaling_coefficient": 1.0,
        "power": 2.0,
        "min_threshold": 0.0
      }
    }
  }
}
```

### 2. Linear Speed Reward
```json
{
  "reward": {
    "variables": {
      "speed": {
        "mode": "linear",
        "base_reward": 0.1,
        "max_speed": 500.0,
        "min_threshold": 0.0
      }
    }
  }
}
```

### 3. Exponential Speed Reward
```json
{
  "reward": {
    "variables": {
      "speed": {
        "mode": "exponential",
        "base_reward": 0.1,
        "max_speed": 500.0,
        "scaling_coefficient": 1.0,
        "min_threshold": 0.0
      }
    }
  }
}
```

### 4. Backward Compatible (Original)
```json
{
  "reward": {
    "variables": {
      "speed": {
        "mode": "binary",
        "op": "greater-than",
        "reference": 1.0,
        "reward": 0.1
      }
    }
  }
}
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | string | "binary" | Reward mode: "binary", "linear", "quadratic", "exponential" |
| `base_reward` | float | 0.1 | Base reward value |
| `max_speed` | float | 500.0 | Maximum expected speed for normalization |
| `scaling_coefficient` | float | 1.0 | Additional scaling factor |
| `power` | float | 2.0 | Power for polynomial scaling (quadratic mode only) |
| `min_threshold` | float | 0.0 | Minimum speed to receive any reward |

## Reward Behavior Analysis

Based on testing with `max_speed = 500.0` and `base_reward = 0.1`:

| Speed | Binary | Linear | Quadratic | Exponential |
|-------|--------|--------|-----------|-------------|
| 0     | 0.000  | 0.000  | 0.000     | 0.000       |
| 100   | 0.100  | 0.020  | 0.004     | 0.013       |
| 200   | 0.100  | 0.040  | 0.016     | 0.029       |
| 300   | 0.100  | 0.060  | 0.036     | 0.048       |
| 400   | 0.100  | 0.080  | 0.064     | 0.071       |
| 500   | 0.100  | 0.100  | 0.100     | 0.100       |

## Migration Steps

### 1. Backup Current Configuration
```bash
cp training.json training_backup.json
```

### 2. Update Configuration
Use the migration script:
```bash
python3 sheeprl/scripts/migrate_reward_config.py
```

Or manually update your `training.json`:
```json
"speed": {
  "mode": "quadratic",
  "base_reward": 0.1,
  "max_speed": 500.0,
  "scaling_coefficient": 1.0,
  "power": 2.0,
  "min_threshold": 0.0
}
```

### 3. Restart Training
The new reward system will take effect immediately upon restart.

## Testing Your Configuration

Use the provided test script to verify behavior:
```bash
python3 sheeprl/simple_reward_test.py
```

## Tuning Recommendations

### For High-Speed Emphasis
- Increase `scaling_coefficient` (e.g., 2.0-3.0)
- Use higher `power` values (e.g., 2.5-3.0)
- Set appropriate `max_speed` based on your track

### For Balanced Learning
- Start with `power = 2.0` and `scaling_coefficient = 1.0`
- Adjust `base_reward` based on other reward components
- Monitor training progress and adjust as needed

### For Conservative Approach
- Use `min_threshold` to only reward meaningful speeds
- Keep `scaling_coefficient` ≤ 1.0
- Consider exponential mode for smoother scaling

## Troubleshooting

### Common Issues
1. **No speed rewards**: Check `min_threshold` and `max_speed` values
2. **Too aggressive rewards**: Reduce `scaling_coefficient` or `power`
3. **Backward compatibility**: Ensure mode is "binary" for original behavior

### Debug Commands
```bash
# Test specific configuration
python3 sheeprl/simple_reward_test.py

# Run unit tests
python3 -m pytest sheeprl/tests/test_envs/test_fzero_reward.py -v
```

## Performance Impact
The enhanced reward system has minimal computational overhead and does not affect training speed or memory usage.

## Files Modified
- `sheeprl/sheeprl/envs/fzero_fixed.py` - Enhanced reward calculation
- `sheeprl/training.json` - Example configuration
- `sheeprl/tests/test_envs/test_fzero_reward.py` - Unit tests
- `sheeprl/scripts/migrate_reward_config.py` - Migration utility