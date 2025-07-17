#!/usr/bin/env python3
"""
AI Content Factory for Short-Form Video

This script uses an LLM to generate a TTS script, a short description, and a
detailed video prompt. It then generates the TTS audio and saves all assets,
including a metadata JSON file, into organized folders.
"""

import torch
import time
import random
import argparse
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer
import soundfile as sf
from kokoro import KPipeline

# --- Comprehensive Master Topic List ---

MASTER_TOPIC_LIST = [
    # Simple, Direct Visuals
    "a beautiful woman walking on a sun-drenched beach",
    "an athletic man jogging through a misty forest at sunrise",
    "a cute golden retriever puppy playing with a red ball in a green park",
    "a sleek supercar driving down a neon-lit city street at night",
    "a majestic eagle soaring over a mountain range",
    "a cozy fireplace with crackling flames",
    "raindrops slowly trickling down a windowpane",
    "a steaming cup of coffee on a wooden table",
    "a skilled chef expertly chopping vegetables",
    "a vibrant coral reef teeming with colorful fish",
    "a stunning woman in a red dress walking through Paris",
    "a powerful wave crashing against a rocky cliff",
    "a field of sunflowers swaying in the breeze",

    # Intellectual & Curiosity-Driven
    "The Fermi Paradox: Where are all the aliens?",
    "Cognitive Dissonance: Why we lie to ourselves",
    "The Great Emu War of Australia",
    "Unusual Deep Sea Creatures and Bioluminescence",
    "The surprisingly complex history of coffee",
    "Stoic Philosophy for modern anxiety",
    "The Carrington Event: The solar storm that almost sent us to the dark ages",
    "The psychology of procrastination and how to beat it",

    # Historical & Mythological
    "Ancient Roman engineering marvels like aqueducts",
    "The mysterious disappearance of the Library of Alexandria",
    "The bizarre story of the Dutch Tulip Mania bubble",
    "The Viking sunstone: A mythical navigation tool",
    "Cleopatra's political genius and charisma",
    "The epic tale of the Trojan Horse",

    # Sci-Fi & Fantasy
    "a futuristic cyberpunk city with flying vehicles",
    "a colossal dragon sleeping on a hoard of gold",
    "an astronaut discovering an alien artifact on Mars",
    "a magical forest with glowing plants and creatures",
    "a massive spaceship entering a wormhole",
    "a knight in shining armor facing a mythical beast",

    # Abstract & Artistic
    "an abstract explosion of colorful ink in water",
    "a time-lapse of a flower blooming",
    "a dynamic dance of light and shadows in a room",
    "geometric patterns shifting and evolving hypnotically",
    "a macro shot of a snowflake forming",
]

# UPDATED: Now includes hashtags in the description prompt.
LLM_META_PROMPT_TEMPLATE = """
You are an AI content creator specializing in short, engaging viral videos. Your task is to generate three pieces of content based on a given topic: a TTS script, a short description, and a detailed video prompt.

**Topic:**
[CHOSEN_TOPIC]

**Task:**

1.  **Write a TTS Script:**
    *   Concise, under 150 words, with a strong hook. Pure narration for a voiceover.
    *   **Crucial Rule:** No sound effects or music cues.

2.  **Write a Short Video Description:**
    *   A catchy, one-sentence description, perfect for a video title or social media post.
    *   Should be under 20 words. Include 1-2 relevant hashtags at the end (e.g., #science #history).

3.  **Write a Video Prompt:**
    *   A single, flowing paragraph under 200 words describing one continuous, uninterrupted camera shot.
    *   Focus on literal, chronological descriptions of action, movement, appearance, and environment.
    *   **Crucial Rule:** No editing terms like 'cut to' or 'final shot'.

**Output Format:**
Provide your response in three distinct parts, clearly separated: "### TTS Script", "### Short Description", and "### Video Prompt".
"""

class AIContentFactory:
    def __init__(self, output_dir: str, interval: int, model_name: str, username: str):
        self.output_dir = Path(output_dir)
        self.interval = interval
        self.username = username
        self.is_running = False
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.tts_pipeline = None
        self.video_counter = 0
        self.user_taste_profile = {}
        self.recently_used_topics = deque(maxlen=5) # Increased to avoid repetition in larger list
        self.output_dir.mkdir(exist_ok=True)

    def _initialize_systems(self):
        """Loads both the LLM and the TTS pipeline."""
        print(f"🧠 Initializing systems...")
        print(f"   - Loading LLM: {self.model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype="auto", device_map="auto"
            )
            print("   ✅ LLM loaded.")
        except Exception as e:
            print(f"❌ Critical Error loading LLM: {e}"); exit()
        
        print("   - Loading TTS Pipeline (Kokoro)...")
        try:
            self.tts_pipeline = KPipeline(lang_code='a') 
            print("   ✅ TTS Pipeline loaded.")
        except Exception as e:
            print(f"❌ Critical Error loading TTS pipeline: {e}"); exit()

    def _run_generation_cycle(self):
        """Orchestrates one full cycle of AI content generation."""
        self.video_counter += 1
        video_id = f"video_{self.video_counter:03d}"
        print(f"--- Cycle {self.video_counter} | ID: {video_id} ---")

        topic, mode = self._select_topic()
        print(f"🔍 Topic selected ({mode.upper()}): {topic}")
        self.recently_used_topics.append(topic)
        
        raw_llm_output = self._query_llm(topic)
        content = self._parse_llm_output(raw_llm_output)
        if not all(content):
            print("❌ Failed to parse all required content. Skipping cycle.")
            self.video_counter -= 1; return

        tts_script, short_desc, video_prompt = content
        print(f"✅ LLM generated all content parts.")
        
        print("🎙️ Generating TTS audio...")
        audio_data = self._generate_tts(tts_script)
        if audio_data is None:
            print("❌ Failed to generate TTS audio. Skipping cycle.")
            self.video_counter -= 1; return
        print("   ✅ TTS audio generated.")

        print(f"💾 Saving asset package to {self.output_dir / video_id}...")
        self._save_assets(video_id, tts_script, short_desc, video_prompt, audio_data)
        print("   ✅ Asset package saved.")

        simulated_watch_percentage = random.uniform(0.15, 0.99)
        print(f"\n📊 Simulating feedback... (Watch %: {simulated_watch_percentage:.2f})")
        self._update_taste_profile(topic, simulated_watch_percentage)

    def _save_assets(self, video_id, tts_script, short_desc, video_prompt, audio_data):
        """Saves all generated assets, including the metadata JSON file."""
        video_path = self.output_dir / video_id
        video_path.mkdir(exist_ok=True)
        
        # Define paths for assets
        final_video_path_str = str(video_path / "video.mp4").replace('\\', '/')
        audio_path = video_path / "audio.wav"

        # Create the metadata dictionary
        metadata = {
            "id": video_id,
            "video_path": final_video_path_str, # Placeholder for the final video
            "description": short_desc,
            "username": self.username
        }
        
        # Write all files
        (video_path / "metadata.json").write_text(json.dumps(metadata, indent=4), encoding='utf-8')
        (video_path / "tts_script.txt").write_text(tts_script, encoding='utf-8')
        (video_path / "video_prompt.txt").write_text(video_prompt, encoding='utf-8')
        sf.write(audio_path, audio_data, 24000)

    def run(self):
        """Main loop to generate content."""
        self._initialize_systems()
        print("\n" + "=" * 50)
        print("🤖 AI Content Factory (TTS + Video + Desc + JSON)")
        print(f"📂 Outputting to: {self.output_dir}")
        print(f"⏰ Interval: {self.interval} seconds")
        print(f"👤 Username: {self.username}")
        print("🛑 Press Ctrl+C to stop")
        print("=" * 50)
        self.is_running = True
        try:
            while self.is_running:
                self._run_generation_cycle()
                print("\n🧠 Current User Taste Profile (Score Descending):")
                if not self.user_taste_profile: print("   (Empty)")
                else:
                    sorted_profile = sorted(self.user_taste_profile.items(), key=lambda i: i[1], reverse=True)
                    for topic, score in sorted_profile: print(f"   - {topic[:50]}: {score:.2f}")
                print(f"\n⏳ Next cycle in {self.interval} seconds...")
                print("-" * 50)
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print(f"\n\n🛑 Stopping factory..."); self.is_running = False

    def _query_llm(self, topic):
        prompt = LLM_META_PROMPT_TEMPLATE.replace("[CHOSEN_TOPIC]", topic)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
        return self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

    def _parse_llm_output(self, raw_output: str) -> tuple:
        try:
            tts_part = raw_output.split("### Short Description")[0].replace("### TTS Script", "").strip()
            desc_part = raw_output.split("### Video Prompt")[0].split("### Short Description")[1].strip()
            prompt_part = raw_output.split("### Video Prompt")[1].strip()
            return tts_part, desc_part, prompt_part
        except IndexError:
            return None, None, None

    def _generate_tts(self, text: str):
        try:
            chunks = [chunk for _, _, chunk in self.tts_pipeline(text, voice='af_heart')]
            return np.concatenate(chunks) if chunks else None
        except Exception as e:
            print(f"   - TTS Generation Error: {e}"); return None

    def _select_topic(self, exploration_rate=0.2):
        if not self.user_taste_profile or random.random() < exploration_rate:
            available = [t for t in MASTER_TOPIC_LIST if t not in self.recently_used_topics] or MASTER_TOPIC_LIST
            return random.choice(available), "explore"
        else:
            candidates = {t: s for t, s in self.user_taste_profile.items() if t not in self.recently_used_topics}
            if not candidates:
                available = [t for t in MASTER_TOPIC_LIST if t not in self.recently_used_topics] or MASTER_TOPIC_LIST
                return random.choice(available), "explore (forced)"
            topics, scores = list(candidates.keys()), list(candidates.values())
            return random.choices(topics, weights=scores, k=1)[0], "exploit"

    def _update_taste_profile(self, topic, watch_percentage):
        score = watch_percentage * 10
        if topic in self.user_taste_profile: self.user_taste_profile[topic] = (self.user_taste_profile[topic] * 0.7) + (score * 0.3)
        else: self.user_taste_profile[topic] = score
        self.user_taste_profile[topic] = round(min(self.user_taste_profile[topic], 10.0), 2)


def main():
    parser = argparse.ArgumentParser(description="AI Content Factory for short-form video.")
    parser.add_argument("--output_dir", default="generated_content", help="Directory to save generated video assets.")
    parser.add_argument("--interval", type=int, default=45, help="Interval in seconds between generation cycles.")
    parser.add_argument("--model_name", default="Qwen/Qwen3-4B", help="Name of the Hugging Face model to use.")
    parser.add_argument("--username", default="ai_content_creations", help="Username to include in the metadata.")
    
    args = parser.parse_args()
    
    factory = AIContentFactory(
        output_dir=args.output_dir,
        interval=args.interval,
        model_name=args.model_name,
        username=args.username
    )
    factory.run()

if __name__ == "__main__":
    main()