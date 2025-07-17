# Random Prompt Generator

This script generates random prompts every 10 seconds and writes them to `prompts.txt` for the `inference_monitor.py` script to process.

## Usage

### Basic Usage
```bash
python prompt_generator.py
```

### With Custom Parameters
```bash
python prompt_generator.py \
  --prompt_file prompts.txt \
  --interval 15 \
  --length 80
```

## Parameters

- `--prompt_file`: Path to the text file to write prompts (default: prompts.txt)
- `--interval`: Interval in seconds between prompt generation (default: 10)
- `--length`: Target length of generated prompts in characters (default: 100)

## Features

- **Realistic Prompts**: Uses common video generation words to create more realistic prompts
- **Random Seeds**: 30% chance to include a random seed with each prompt
- **Configurable**: Adjust interval and prompt length
- **Safe**: Creates the prompt file if it doesn't exist
- **Clean Output**: Shows generation progress with timestamps

## Example Output

```
🎲 Random Prompt Generator for LTX-Video
==================================================
📄 Writing to: prompts.txt
⏰ Interval: 10 seconds
📏 Prompt length: ~100 characters
🛑 Press Ctrl+C to stop
==================================================
📝 Generated prompt: 'Beautiful woman dancing on beach' with seed 123456
📊 [14:30:15] Total prompts generated: 1
------------------------------
📝 Generated prompt: 'Amazing sunset over mountains'
📊 [14:30:25] Total prompts generated: 2
------------------------------
```

## Integration with Monitor

This script works perfectly with `inference_monitor.py`:

1. **Terminal 1**: Start the monitor
   ```bash
   python inference_monitor.py
   ```

2. **Terminal 2**: Start the prompt generator
   ```bash
   python prompt_generator.py
   ```

The monitor will automatically detect new prompts and generate videos!

## Prompt Format

The generator creates prompts in these formats:
- `Beautiful woman dancing on beach`
- `Amazing sunset over mountains|123456` (with random seed)

Both formats are compatible with the video generation monitor. 