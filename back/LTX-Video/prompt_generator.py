#!/usr/bin/env python3
"""
AI Content Factory for Short-Form Video (with Enhanced Subtitles)

This script uses an LLM to generate all text assets, then creates a voice-over
and a corresponding high-quality SRT subtitle file with corrected timestamps and
consolidated words/punctuation for better readability.
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

# --- Configuration Constants ---

MASTER_TOPIC_LIST = [
    # Simple, Direct Visuals
    "car crash",
    "people fighting",
    "people kissing",
    "people hugging",
    "people crying",
    "people laughing",
    "indian street food vendor",
    "woman in a red dress",
    "man in a business suit with rolexes and champagne",
    "woman in a bikini",
    "blonde woman in a green dress with a big ass, dancing",
    "podcast host talking in a microphone, surprised, laughing",
    "black man dancing to music",
    "black woman in a blue dress, dancing",
    "black woman in a yellow bikini, dancing",
    "korean guy eating ramen",
    "korean mukbang",
    "k pop artist",
    "k pop group",
    "k pop song",
    "very cute black cats",
    "very cute white cats",
    "very cute black dogs",
    "very cute white dogs",
    "very cute tigers",
    "very cute lions",
    "very cute leopards",
    "very cute cheetah",
    "very cute panda",
    "football celebration goal",
    "basketball dunk",
    "baseball home run",
    "tennis player hitting a tennis ball and then having a trophy",
    "woman in a red dress, dancing",
    "woman in a yellow dress, dancing",
    "woman in a blue dress, dancing",
    "woman in a green dress, dancing",
    "woman in a pink dress, dancing",
    "woman in a purple dress, dancing",
]

KOKORO_VOICES = [
    'af_heart', 'af_alloy', 'af_aoede', 'af_bella', 'af_jessica', 'af_kore', 'af_nova',
    'af_river', 'af_sarah', 'af_sky', 'am_adam', 'am_echo', 'am_eric', 'am_fenrir',
    'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa'
]

PUNCTUATION_SET = {'.', ',', '?', '!', ';', ':'}
MIN_SUBTITLE_DURATION_S = 0.1 # 100ms

LLM_META_PROMPT_TEMPLATE = """
You are an AI content creator specializing in short, engaging viral videos. The videos are quite simple, they are made for a 25 years old white cis person with a short attention span. He is doomscrolling on this generated TikTok. Your tiktok usually have twists that surprise the audience. Your task is to generate four pieces of content based on a given topic: a TTS script, a short description, a username, and a detailed video prompt.

**Topic:**
[CHOSEN_TOPIC]

**Task:**

1.  **Write a TTS Script:**
    *   Concise, under 60 words, with a strong hook. Pure narration for a voiceover.
    *   **Crucial Rule:** No sound effects or music cues.

2.  **Write a Short Video Description:**
    *   A catchy, one-sentence description, perfect for a video title or social media post.
    *   Should be under 20 words. Optionnaly include 1-2 relevant hashtags at the end (e.g., #science #history).

3.  **Write a Username:**
    *   A catchy, memorable username that fits the content style.
    *   Should be 1-3 words, no spaces, can include numbers or underscores.

4.  **Write a Video Prompt:**
    *   The prompt is the peak of the story. It should be literally the most shocking and clickbait-y thing of the video. Usually the twist.  
    *   A single, flowing paragraph under 100 words describing one continuous, uninterrupted camera shot.
    *   Focus on literal descriptions of action, movement, appearance, and environment.
    *   **Crucial Rule:** No editing terms like 'cut to' or 'final shot'.

**Output Format:**
Provide your response in four distinct parts, clearly separated: "### TTS Script", "### Short Description", "### Username", and "### Video Prompt".
"""

class AIContentFactory:
    def __init__(self, output_dir: str, interval: int, model_name: str):
        self.output_dir = Path(output_dir)
        self.interval = interval
        self.is_running = False
        self.model_name = model_name
        self.tokenizer, self.model, self.tts_pipeline = None, None, None
        self.user_taste_profile = {}
        self.recently_used_topics = deque(maxlen=5)
        self.output_dir.mkdir(exist_ok=True)
        
        # Detect existing videos and continue from there
        existing_videos = sorted([d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith('video_')])
        if existing_videos:
            # Get the highest video number
            max_video_num = max([int(d.name.split('_')[1]) for d in existing_videos])
            self.video_counter = max_video_num
            self.processed_videos = set()  # Track which videos have been processed for watch stats
            print(f"📁 Found {len(existing_videos)} existing videos, continuing from video_{self.video_counter + 1:03d}")
        else:
            self.video_counter = 0
            self.processed_videos = set()
            print("📁 Starting fresh - no existing videos found")

    def _initialize_systems(self):
        print(f"🧠 Initializing systems...")
        print(f"   - Loading LLM: {self.model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", device_map="auto")
            print("   ✅ LLM loaded.")
        except Exception as e: print(f"❌ Critical Error loading LLM: {e}"); exit()
        
        print("   - Loading TTS Pipeline (Kokoro)...")
        try:
            self.tts_pipeline = KPipeline(lang_code='a', repo_id="hexgrad/Kokoro-82M") 
            print("   ✅ TTS Pipeline loaded.")
        except Exception as e: print(f"❌ Critical Error loading TTS pipeline: {e}"); exit()

    def _process_existing_watch_stats(self):
        """Process all existing watch stats to build initial taste profile."""
        print("📊 Processing existing watch statistics...")
        
        # Get all video folders in order
        video_folders = sorted([d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith('video_')])
        
        processed_count = 0
        for video_folder in video_folders:
            try:
                video_num = int(video_folder.name.split('_')[1])
                watch_stats_file = video_folder / "watch_stats.txt"
                
                if watch_stats_file.exists():
                    print(f"   📈 Processing watch stats for video_{video_num:03d}")
                    
                    # Read watch percentage from file
                    try:
                        watch_percentage = float(watch_stats_file.read_text().strip())
                        print(f"      Watch percentage: {watch_percentage:.2f}")
                    except ValueError:
                        print(f"      ⚠️ Invalid watch percentage in {watch_stats_file}")
                        continue
                    
                    # Read metadata to get topic index
                    metadata_file = video_folder / "metadata.json"
                    if metadata_file.exists():
                        try:
                            metadata = json.loads(metadata_file.read_text())
                            topic_index = metadata.get("topic_index")
                            if topic_index is not None:
                                topic = MASTER_TOPIC_LIST[topic_index]
                                print(f"      Topic: {topic}")
                                self._update_taste_profile(topic, watch_percentage)
                                processed_count += 1
                                self.processed_videos.add(video_num)  # Mark as processed
                            else:
                                print(f"      ⚠️ No topic_index found in metadata")
                        except (json.JSONDecodeError, IndexError) as e:
                            print(f"      ⚠️ Error reading metadata: {e}")
                    else:
                        print(f"      ⚠️ No metadata.json found")
                    
                else:
                    print(f"   ⏳ No watch stats for video_{video_num:03d} (not watched yet)")
                    
            except (ValueError, IndexError) as e:
                print(f"   ⚠️ Error processing folder {video_folder.name}: {e}")
                continue
        
        print(f"   ✅ Processed {processed_count} existing watch statistics")
        if self.user_taste_profile:
            print("   🧠 Current taste profile:")
            sorted_profile = sorted(self.user_taste_profile.items(), key=lambda i: i[1], reverse=True)
            for topic, score in sorted_profile[:5]:  # Show top 5
                print(f"      - {topic[:50]}: {score:.2f}")

    def _check_and_update_watch_stats(self):
        """Check for new watch_stats.txt files and update taste profile accordingly."""
        print("📊 Checking for new watch statistics...")
        
        # Get all video folders in order
        video_folders = sorted([d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith('video_')])
        
        new_stats_found = False
        for video_folder in video_folders:
            try:
                video_num = int(video_folder.name.split('_')[1])
                if video_num in self.processed_videos:
                    continue  # Already processed
                    
                watch_stats_file = video_folder / "watch_stats.txt"
                if watch_stats_file.exists():
                    print(f"   📈 Found watch stats for video_{video_num:03d}")
                    
                    # Read watch percentage from file
                    try:
                        watch_percentage = float(watch_stats_file.read_text().strip())
                        print(f"      Watch percentage: {watch_percentage:.2f}")
                    except ValueError:
                        print(f"      ⚠️ Invalid watch percentage in {watch_stats_file}")
                        continue
                    
                    # Read metadata to get topic index
                    metadata_file = video_folder / "metadata.json"
                    if metadata_file.exists():
                        try:
                            metadata = json.loads(metadata_file.read_text())
                            topic_index = metadata.get("topic_index")
                            if topic_index is not None:
                                topic = MASTER_TOPIC_LIST[topic_index]
                                print(f"      Topic: {topic}")
                                self._update_taste_profile(topic, watch_percentage)
                                new_stats_found = True
                                self.processed_videos.add(video_num)  # Mark as processed
                            else:
                                print(f"      ⚠️ No topic_index found in metadata")
                        except (json.JSONDecodeError, IndexError) as e:
                            print(f"      ⚠️ Error reading metadata: {e}")
                    else:
                        print(f"      ⚠️ No metadata.json found")
                    
                else:
                    print(f"   ⏳ No watch stats for video_{video_num:03d} (not watched yet)")
                    
            except (ValueError, IndexError) as e:
                print(f"   ⚠️ Error processing folder {video_folder.name}: {e}")
                continue
        
        if not new_stats_found:
            print("   📊 No new watch statistics found")
        else:
            print("   ✅ Watch statistics updated")

    def _run_generation_cycle(self):
        # First, check for new watch stats before generating new content
        self._check_and_update_watch_stats()
        
        self.video_counter += 1
        video_id = f"video_{self.video_counter:03d}"
        print(f"--- Cycle {self.video_counter} | ID: {video_id} ---")

        topic, mode = self._select_topic()
        topic_index = MASTER_TOPIC_LIST.index(topic)
        print(f"🔍 Topic selected ({mode.upper()}): {topic} (index: {topic_index})")
        self.recently_used_topics.append(topic)
        
        raw_llm_output = self._query_llm(topic)
        content = self._parse_llm_output(raw_llm_output)
        if not all(content):
            print("❌ Failed to parse content from LLM. Skipping."); self.video_counter -= 1; return

        tts_script, short_desc, video_prompt, username = content
        print(f"✅ LLM generated all content parts (Username: {username}).")
        
        chosen_voice = random.choice(KOKORO_VOICES)
        print(f"🎙️ Generating TTS audio & timestamps with voice: '{chosen_voice}'...")
        audio_data, raw_timestamps = self._generate_tts(tts_script, voice_name=chosen_voice)
        if audio_data is None:
            print("❌ Failed to generate TTS. Skipping."); self.video_counter -= 1; return
        print("   ✅ TTS audio and raw timestamps generated.")

        print(f"✍️ Consolidating subtitles for readability...")
        consolidated_timestamps = self._consolidate_subtitle_tokens(raw_timestamps)
        print(f"   ✅ Subtitles consolidated.")
        
        print(f"✍️ Creating SRT subtitle file...")
        srt_content = self._create_srt_content(consolidated_timestamps)
        print("   ✅ SRT content created.")

        print(f"💾 Saving asset package to {self.output_dir / video_id}...")
        self._save_assets(video_id, tts_script, short_desc, video_prompt, audio_data, chosen_voice, srt_content, username, topic_index)
        print("   ✅ Asset package saved.")

    def _save_assets(self, video_id, tts_script, short_desc, video_prompt, audio_data, voice_used, srt_content, username, topic_index):
        video_path = self.output_dir / video_id
        video_path.mkdir(exist_ok=True)
        final_video_path_str = str(video_path / "video.mp4").replace('\\', '/')
        subtitle_path_str = str(video_path / "subtitles.srt").replace('\\', '/')

        metadata = {
            "id": video_id, "username": username, "description": short_desc,
            "voice": voice_used, "video_path": final_video_path_str, "subtitle_path": subtitle_path_str,
            "topic_index": topic_index
        }
        
        (video_path / "metadata.json").write_text(json.dumps(metadata, indent=4), encoding='utf-8')
        (video_path / "tts_script.txt").write_text(tts_script, encoding='utf-8')
        (video_path / "video_prompt.txt").write_text(video_prompt, encoding='utf-8')
        (video_path / "subtitles.srt").write_text(srt_content, encoding='utf-8')
        sf.write(video_path / "audio.wav", audio_data, 24000)

    def run(self):
        self._initialize_systems()
        self._process_existing_watch_stats() # Process existing stats at startup
        print("\n" + "=" * 50)
        print("🤖 AI Content Factory (with Enhanced Subtitles)")
        print(f"📂 Outputting to: {self.output_dir}")
        print(f"⏰ Interval: {self.interval} seconds")
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
                print(f"\n⏳ Next cycle in {self.interval} seconds..."); time.sleep(self.interval)
                print("-" * 50)
        except KeyboardInterrupt: print(f"\n\n🛑 Stopping factory..."); self.is_running = False

    def _format_srt_timestamp(self, seconds: float) -> str:
        total_seconds, milliseconds = int(seconds), int((seconds * 1000) % 1000)
        hours, minutes, seconds_part = total_seconds // 3600, (total_seconds % 3600) // 60, total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds_part:02d},{milliseconds:03d}"
    
    # =============================================================
    # == THIS IS THE MODIFIED/FIXED METHOD ==
    # =============================================================
    def _consolidate_subtitle_tokens(self, tokens: list) -> list:
        """Merges punctuation and very short words into the previous token for better subtitle readability."""
        if not tokens:
            return []

        # Filter for valid tokens first to simplify the main loop
        valid_raw_tokens = [t for t in tokens if t.text.strip() and t.start_ts is not None and t.end_ts is not None]
        if not valid_raw_tokens:
            return []
            
        consolidated_tokens = [valid_raw_tokens[0]]

        for current_token in valid_raw_tokens[1:]:
            previous_token = consolidated_tokens[-1]
            
            duration = current_token.end_ts - current_token.start_ts
            current_text = current_token.text.strip()
            is_punctuation = current_text in PUNCTUATION_SET
            is_too_short = duration < MIN_SUBTITLE_DURATION_S

            if is_punctuation or is_too_short:
                # === MODIFIED LOGIC HERE ===
                # Add a space only if merging a word, not punctuation.
                if is_punctuation:
                    previous_token.text += current_text
                else: # It's a short word
                    previous_token.text += " " + current_text
                # === END OF MODIFIED LOGIC ===

                # Always update the end time to encompass the merged token
                previous_token.end_ts = current_token.end_ts
            else:
                consolidated_tokens.append(current_token)

        return consolidated_tokens
    # =============================================================
    # =================== END OF MODIFIED SECTION =================
    # =============================================================

    def _create_srt_content(self, tokens: list) -> str:
        srt_blocks = []
        for i, token in enumerate(tokens, 1):
            if not token.text.strip() or token.start_ts is None or token.end_ts is None: continue
            start = self._format_srt_timestamp(token.start_ts)
            end = self._format_srt_timestamp(token.end_ts)
            srt_blocks.append(f"{i}\n{start} --> {end}\n{token.text.strip()}")
        return "\n\n".join(srt_blocks)

    def _generate_tts(self, text: str, voice_name: str):
        try:
            audio_chunks, all_tokens = [], []
            cumulative_duration_s, sample_rate = 0.0, 24000
            generator = self.tts_pipeline(text, voice=voice_name)
            for result in generator:
                audio_chunk_np, current_tokens = result.audio.cpu().numpy(), result.tokens
                if cumulative_duration_s > 0:
                    for token in current_tokens:
                        if token.start_ts is not None: token.start_ts += cumulative_duration_s
                        if token.end_ts is not None: token.end_ts += cumulative_duration_s
                all_tokens.extend(current_tokens)
                audio_chunks.append(audio_chunk_np)
                cumulative_duration_s += len(audio_chunk_np) / sample_rate
            if not audio_chunks: return None, None
            return np.concatenate(audio_chunks), all_tokens
        except Exception as e: print(f"   - TTS Generation Error: {e}"); return None, None

    def _query_llm(self, topic):
        prompt = LLM_META_PROMPT_TEMPLATE.replace("[CHOSEN_TOPIC]", topic)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
        return self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

    def _parse_llm_output(self, raw_output: str):
        try:
            parts = {}
            chunks = [chunk for chunk in raw_output.split('### ') if chunk]
            for chunk in chunks:
                header, content = chunk.split('\n', 1)
                parts[header.strip()] = content.strip()
            tts_script, short_desc = parts.get("TTS Script"), parts.get("Short Description")
            video_prompt, username = parts.get("Video Prompt"), parts.get("Username")
            if username: username = username.split()[0].lower().strip()
            return tts_script, short_desc, video_prompt, username
        except (ValueError, KeyError): return None, None, None, None

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
    parser = argparse.ArgumentParser(description="AI Content Factory for short-form video with enhanced subtitles.")
    parser.add_argument("--output_dir", default="generated_content", help="Directory to save generated video assets.")
    parser.add_argument("--interval", type=int, default=10, help="Interval in seconds between generation cycles.")
    parser.add_argument("--model_name", default="Qwen/Qwen3-4B", help="Name of the Hugging Face model to use.")
    args = parser.parse_args()
    factory = AIContentFactory(**vars(args))
    factory.run()

if __name__ == "__main__":
    main()
