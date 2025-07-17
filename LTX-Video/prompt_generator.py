#!/usr/bin/env python3
"""
Random Prompt Generator for LTX-Video

This script generates random prompts every 10 seconds and writes them to prompts.txt
for the inference_monitor.py script to process.
"""

import time
import random
import string
import argparse
from datetime import datetime
from pathlib import Path


class RandomPromptGenerator:
    def __init__(self, prompt_file: str = "prompts.txt", interval: int = 10, prompt_length: int = 100):
        self.prompt_file = prompt_file
        self.interval = interval
        self.prompt_length = prompt_length
        self.is_running = False
        
        # Create prompt file if it doesn't exist
        Path(self.prompt_file).touch(exist_ok=True)
        
    def _generate_random_prompt(self) -> str:
        """Generate a random prompt of specified length."""
        # Define character sets for more realistic prompts
        letters = string.ascii_lowercase
        spaces = " "
        punctuation = ",.!?"
        
        # Create a more realistic prompt structure
        words = []
        current_length = 0
        
        # Common video generation words
        video_words = [
            "beautiful", "amazing", "stunning", "gorgeous", "magnificent",
            "woman", "man", "person", "people", "child", "baby",
            "dancing", "running", "walking", "jumping", "swimming",
            "beach", "mountain", "forest", "city", "street", "park",
            "sunset", "sunrise", "night", "day", "morning", "evening",
            "cat", "dog", "bird", "fish", "horse", "elephant",
            "car", "bike", "train", "plane", "boat", "ship",
            "flower", "tree", "grass", "water", "fire", "smoke",
            "dress", "shirt", "hat", "shoes", "jewelry",
            "smiling", "laughing", "crying", "thinking", "working",
            "cooking", "eating", "drinking", "reading", "writing",
            "playing", "singing", "music", "art", "painting", "drawing"
        ]
        
        # Generate a realistic prompt
        num_words = random.randint(3, 8)
        for _ in range(num_words):
            if random.random() < 0.7:  # 70% chance to use a video word
                word = random.choice(video_words)
            else:
                # Generate a random word
                word_length = random.randint(3, 8)
                word = ''.join(random.choice(letters) for _ in range(word_length))
            
            words.append(word)
            
        # Add some punctuation occasionally
        if random.random() < 0.3:
            words.append(random.choice(punctuation))
            
        prompt = " ".join(words)
        
        # Ensure the prompt is close to the desired length
        while len(prompt) < self.prompt_length * 0.8:
            word = random.choice(video_words)
            prompt += " " + word
            
        # Truncate if too long
        if len(prompt) > self.prompt_length:
            prompt = prompt[:self.prompt_length].rsplit(' ', 1)[0]
            
        return prompt.capitalize()
    
    def _write_prompt_to_file(self, prompt: str):
        """Write the prompt to the file with optional seed."""
        # Randomly decide if we want to include a seed
        include_seed = random.random() < 0.3  # 30% chance to include seed
        
        if include_seed:
            seed = random.randint(1, 999999)
            line = f"{prompt}|{seed}\n"
        else:
            line = f"{prompt}\n"
            
        with open(self.prompt_file, "a") as f:
            f.write(line)
            
        print(f"📝 Generated prompt: '{prompt}'" + (f" with seed {seed}" if include_seed else ""))
    
    def run(self):
        """Main loop to generate prompts."""
        print("🎲 Random Prompt Generator for LTX-Video")
        print("=" * 50)
        print(f"📄 Writing to: {self.prompt_file}")
        print(f"⏰ Interval: {self.interval} seconds")
        print(f"📏 Prompt length: ~{self.prompt_length} characters")
        print("🛑 Press Ctrl+C to stop")
        print("=" * 50)
        
        self.is_running = True
        prompt_count = 0
        
        try:
            while self.is_running:
                # Generate and write prompt
                prompt = self._generate_random_prompt()
                self._write_prompt_to_file(prompt)
                
                prompt_count += 1
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"📊 [{current_time}] Total prompts generated: {prompt_count}")
                print("-" * 30)
                
                # Wait for next generation
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping generator...")
            print(f"📊 Total prompts generated: {prompt_count}")
            print("👋 Goodbye!")
            self.is_running = False


def main():
    parser = argparse.ArgumentParser(description="Generate random prompts for LTX-Video")
    parser.add_argument("--prompt_file", default="prompts.txt", help="Path to the prompt file")
    parser.add_argument("--interval", type=int, default=10, help="Interval in seconds between prompts")
    parser.add_argument("--length", type=int, default=100, help="Target length of generated prompts")
    
    args = parser.parse_args()
    
    generator = RandomPromptGenerator(
        prompt_file=args.prompt_file,
        interval=args.interval,
        prompt_length=args.length
    )
    
    generator.run()


if __name__ == "__main__":
    main() 