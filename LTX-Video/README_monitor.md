# Video Generation Monitor

This script monitors a text file for new prompts and automatically generates videos based on them.

## Usage

### Basic Usage
```bash
python inference_monitor.py
```

### With Custom Parameters
```bash
python inference_monitor.py \
  --prompt_file prompts.txt \
  --check_interval 2.0 \
  --pipeline_config configs/ltxv-2b-0.9.8-distilled.yaml \
  --height 768 \
  --width 768 \
  --num_frames 240 \
  --seed 171198 \
  --frame_rate 30
```

## Prompt File Format

The script monitors a text file (default: `prompts.txt`) for new prompts. Each line should contain:

- **Simple format**: Just the prompt
  ```
  naked woman on the beach
  beautiful sunset over mountains
  ```

- **With custom seed**: Prompt followed by `|` and the seed number
  ```
  naked woman on the beach|42
  beautiful sunset over mountains|123
  ```

## How It Works

1. The script initializes the pipeline once at startup
2. It monitors the prompt file every `check_interval` seconds (default: 1 second)
3. When new lines are detected, they are added to a generation queue
4. Videos are generated one at a time from the queue
5. Output videos are saved to the output directory with unique filenames

## Features

- **Queue-based processing**: Multiple prompts are queued and processed sequentially
- **Duplicate detection**: Already processed prompts are ignored
- **Custom seeds**: Each prompt can have its own seed value
- **Thread-safe**: Uses locks to handle concurrent file reading
- **Graceful shutdown**: Press Ctrl+C to stop the monitor

## Parameters

- `--prompt_file`: Path to the text file containing prompts (default: prompts.txt)
- `--check_interval`: Interval in seconds to check for new prompts (default: 1.0)
- `--pipeline_config`: Path to the pipeline config file
- `--seed`: Default random seed for inference
- `--height`: Height of the output video frames
- `--width`: Width of the output video frames
- `--num_frames`: Number of frames to generate
- `--frame_rate`: Frame rate for the output video
- `--output_path`: Path to save output videos
- `--negative_prompt`: Negative prompt for undesired features
- `--offload_to_cpu`: Offload to CPU for memory efficiency

## Example Workflow

1. Start the monitor:
   ```bash
   python inference_monitor.py
   ```

2. Add prompts to `prompts.txt`:
   ```
   naked woman on the beach
   beautiful sunset over mountains|42
   cat playing with yarn|123
   ```

3. The script will automatically detect and process each prompt, generating videos in the output directory.

4. Stop the monitor with Ctrl+C when done. 