#!/usr/bin/env python3
"""
AI Content Factory for Short-Form Video (with Intelligent Mixing)

This script automates the entire content creation pipeline:
1. Downloads the ACEStep music model if not present.
2. Uses an LLM to generate all text and music prompts.
3. Creates a voice-over and a background music track.
4. Intelligently mixes the audio by normalizing both tracks to target loudness levels.
5. Generates a corresponding VTT subtitle file with custom styling.
"""
import torch
import time
import random
import argparse
import numpy as np
import json
import os
from pathlib import Path
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer
import soundfile as sf
from kokoro import KPipeline

# --- Imports for Auto-Download and Audio Mixing ---
from huggingface_hub import snapshot_download
from pydub import AudioSegment

# --- ACEStep Music Generation ---
from acestep.pipeline_ace_step import ACEStepPipeline


# --- Configuration Constants ---

# --- NEW: Intelligent Mixing Targets ---
# The voice will be normalized to this level. A good standard for clear speech.
TARGET_VOICE_DBFS = -20.0
# The music will be normalized to this level. Should be significantly lower than the voice.
TARGET_MUSIC_DBFS = -25.0

MASTER_TOPIC_LIST = [
    # ... (list remains the same)
    "car crash", "people fighting", "people kissing", "people hugging", "people crying", "people laughing", "indian street food vendor", "woman in a red dress", "man in a business suit with rolexes and champagne", "woman in a bikini", "blonde woman in a green dress with a big ass, dancing", "podcast host talking in a microphone, surprised, laughing", "black man dancing to music", "black woman in a blue dress, dancing", "black woman in a yellow bikini, dancing", "korean guy eating ramen", "korean mukbang", "k pop artist", "k pop group", "k pop song", "very cute black cats", "very cute white cats", "very cute black dogs", "very cute white dogs", "very cute tigers", "very cute lions", "very cute leopards", "very cute cheetah", "very cute panda", "football celebration goal", "basketball dunk", "baseball home run", "tennis player hitting a tennis ball and then having a trophy", "woman in a red dress, dancing", "woman in a yellow dress, dancing", "woman in a blue dress, dancing", "woman in a green dress, dancing", "woman in a pink dress, dancing", "woman in a purple dress, dancing",
]

KOKORO_VOICES = [
    'af_heart', 'af_alloy', 'af_aoede', 'af_bella', 'af_jessica', 'af_kore', 'af_nova', 'af_river', 'af_sarah', 'af_sky', 'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa'
]

PUNCTUATION_SET = {'.', ',', '?', '!', ';', ':'}
MIN_SUBTITLE_DURATION_S = 0.1

LLM_META_PROMPT_TEMPLATE = """
You are an AI content creator specializing in short, engaging viral videos. The videos are quite simple, they are made for a 25 years old white cis person with a short attention span. He is doomscrolling on this generated TikTok. Your tiktok usually have twists that surprise the audience. Your task is to generate five pieces of content based on a given topic: a TTS script, a short description, a username, a video prompt, and a music prompt.

**Topic:**
[CHOSEN_TOPIC]

**Task:**

1.  **Write a TTS Script:**
    *   Concise, under 60 words, with a strong hook. Pure narration for a voiceover.
    *   **Crucial Rule:** No sound effects or music cues.

2.  **Write a Short Video Description:**
    *   A catchy, one-sentence description, perfect for a video title or social media post.
    *   Should be under 20 words. Optionally include 1-2 relevant hashtags at the end (e.g., #science #history).

3.  **Write a Username:**
    *   A catchy, memorable username that fits the content style.
    *   Should be 1-3 words, no spaces, can include numbers or underscores.

4.  **Write a Video Prompt:**
    *   The prompt is the peak of the story. It should be literally the most shocking and clickbait-y thing of the video. Usually the twist.
    *   A single, flowing paragraph under 100 words describing one continuous, uninterrupted camera shot.
    *   Focus on literal descriptions of action, movement, appearance, and environment.
    *   **Crucial Rule:** No editing terms like 'cut to' or 'final shot'.

5.  **Write a Music Prompt:**
    *   A prompt for a generative music AI (ACEStep).
    *   Describe the mood, genre, and instrumentation. Use comma-separated keywords.
    *   Example: "lo-fi, chillhop, relaxing, vinyl crackle, soft piano, mellow drums, 80 BPM"
    *   Another Example: "epic, cinematic, orchestral, intense, powerful drums, choir, suspenseful, 140 BPM"

**Output Format:**
Provide your response in five distinct parts, clearly separated: "### TTS Script", "### Short Description", "### Username", "### Video Prompt", and "### Music Prompt".
"""
class AIContentFactory:
    def __init__(self, output_dir: str, interval: int, model_name: str, acestep_checkpoint_path: str, bf16: bool, cpu_offload: bool):
        self.output_dir = Path(output_dir)
        self.interval = interval
        self.is_running = False
        
        # Model configurations
        self.model_name = model_name
        self.acestep_checkpoint_path = Path(acestep_checkpoint_path)
        self.use_bf16 = bf16
        self.use_cpu_offload = cpu_offload
        self._ensure_acestep_checkpoints()

        # Pipelines
        self.tokenizer, self.model, self.tts_pipeline, self.music_pipeline = None, None, None, None
        
        self.user_taste_profile = {}
        self.recently_used_topics = deque(maxlen=5)
        self.output_dir.mkdir(exist_ok=True)
        
        existing_videos = sorted([d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith('video_')])
        self.video_counter = max([int(d.name.split('_')[1]) for d in existing_videos]) if existing_videos else 0
        self.processed_videos = set()
        print(f"📁 Found {len(existing_videos)} existing videos, continuing from video_{self.video_counter + 1:03d}")

    def _ensure_acestep_checkpoints(self):
        """Checks for ACEStep checkpoints and downloads them if they don't exist."""
        if not self.acestep_checkpoint_path.exists():
            print(f"🎵 ACEStep checkpoint not found at '{self.acestep_checkpoint_path}'.")
            print("   Downloading from Hugging Face Hub (ACE-Step/ACE-Step-v1-3.5B)... This may take a moment.")
            try:
                snapshot_download(repo_id="ACE-Step/ACE-Step-v1-3.5B", local_dir=self.acestep_checkpoint_path, local_dir_use_symlinks=False)
                print("   ✅ ACEStep checkpoint downloaded successfully.")
            except Exception as e:
                print(f"❌ Critical Error downloading ACEStep model: {e}"); exit()
        else:
            print(f"🎵 Found existing ACEStep checkpoint at '{self.acestep_checkpoint_path}'.")

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
        
        print("   - Loading Music Pipeline (ACEStep)...")
        try:
            self.music_pipeline = ACEStepPipeline(checkpoint_dir=str(self.acestep_checkpoint_path), dtype="bfloat16" if self.use_bf16 else "float32", cpu_offload=self.use_cpu_offload, torch_compile=True)
            print("   ✅ Music Pipeline loaded.")
        except Exception as e: print(f"❌ Critical Error loading ACEStep pipeline: {e}"); exit()

    def _run_generation_cycle(self):
        self._check_and_update_watch_stats()
        
        self.video_counter += 1
        video_id = f"video_{self.video_counter:03d}"
        video_path = self.output_dir / video_id
        video_path.mkdir(exist_ok=True)
        print(f"--- Cycle {self.video_counter} | ID: {video_id} ---")

        topic, mode = self._select_topic()
        topic_index = MASTER_TOPIC_LIST.index(topic)
        print(f"🔍 Topic selected ({mode.upper()}): {topic}")
        self.recently_used_topics.append(topic)
        
        raw_llm_output = self._query_llm(topic)
        content = self._parse_llm_output(raw_llm_output)
        if not all(content):
            print("❌ Failed to parse content from LLM. Skipping."); self.video_counter -= 1; return
        tts_script, short_desc, video_prompt, username, music_prompt = content
        print(f"✅ LLM generated all content parts.")
        
        print(f"🎙️ Generating TTS audio...")
        audio_data, raw_timestamps, audio_duration = self._generate_tts(tts_script, voice_name=random.choice(KOKORO_VOICES))
        if audio_data is None:
            print("❌ Failed to generate TTS. Skipping."); self.video_counter -= 1; return
        tts_audio_path = video_path / "tts_audio.wav"
        sf.write(tts_audio_path, audio_data, 24000)
        print(f"   ✅ TTS audio generated ({audio_duration:.2f}s).")

        print(f"🎵 Generating music...")
        music_audio_path = self._generate_music(music_prompt, audio_duration, video_path)
        if not music_audio_path:
             print("❌ Failed to generate music. Skipping."); self.video_counter -= 1; return
        print("   ✅ Music track generated.")
        
        print("🎧 Intelligently mixing TTS and music...")
        final_audio_path = video_path / "final_audio.wav"
        mix_success = self._mix_audio_intelligently(tts_audio_path, music_audio_path, final_audio_path)
        if not mix_success:
            print("❌ Failed to mix audio. Skipping."); self.video_counter -= 1; return
        print("   ✅ Audio mixed successfully.")
        
        print(f"✍️ Creating VTT subtitles...")
        consolidated_timestamps = self._consolidate_subtitle_tokens(raw_timestamps)
        vtt_content = self._create_vtt_content(consolidated_timestamps)
        print("   ✅ VTT subtitles created.")

        print(f"💾 Saving asset package to {video_path}...")
        self._save_assets(video_id, video_path, tts_script, short_desc, video_prompt, music_prompt, vtt_content, username, topic_index)
        print("   ✅ Asset package saved.")

    def _mix_audio_intelligently(self, tts_path: Path, music_path: Path, output_path: Path) -> bool:
        """
        Mixes audio by normalizing both tracks to target loudness levels for consistent, clear results.
        """
        try:
            voice = AudioSegment.from_wav(tts_path)
            music = AudioSegment.from_wav(music_path)

            # 1. Calculate the gain needed to normalize each track to its target
            voice_gain = TARGET_VOICE_DBFS - voice.dBFS
            music_gain = TARGET_MUSIC_DBFS - music.dBFS

            # 2. Apply the gain
            normalized_voice = voice.apply_gain(voice_gain)
            normalized_music = music.apply_gain(music_gain)
            
            # 3. Ensure music is long enough, looping if necessary
            if len(normalized_music) < len(normalized_voice):
                times_to_loop = (len(normalized_voice) // len(normalized_music)) + 1
                normalized_music = normalized_music * times_to_loop
            
            # 4. Trim music to voice length and add a fade-out
            background_music = normalized_music[:len(normalized_voice)].fade_out(1500)

            # 5. Overlay the normalized voice onto the normalized background music
            mixed_audio = background_music.overlay(normalized_voice)

            mixed_audio.export(output_path, format="wav")
            return True
        except Exception as e:
            print(f"   - ❌ An unexpected error occurred during audio mixing: {e}")
            return False

    def _generate_music(self, prompt: str, duration: float, save_dir: Path) -> Path | None:
        try:
            for old_file in save_dir.glob("output_*.wav"): old_file.unlink()
            self.music_pipeline(prompt=prompt, audio_duration=int(duration) + 2, lyrics="", save_path=str(save_dir), infer_step=60, guidance_scale=15, scheduler_type="euler", manual_seeds="-1")
            generated_files = list(save_dir.glob("output_*.wav"))
            if generated_files:
                music_path = save_dir / "music.wav"
                generated_files[0].rename(music_path)
                return music_path
            return None
        except Exception as e:
            print(f"   - ❌ Music Generation Error: {e}"); return None

    def _save_assets(self, video_id, video_path, tts_script, short_desc, video_prompt, music_prompt, vtt_content, username, topic_index):
        metadata = {
            "id": video_id, "username": username, "description": short_desc,
            "video_path": str(video_path / "video.mp4").replace('\\', '/'),
            "subtitle_path": str(video_path / "subtitles.vtt").replace('\\', '/'),
            "final_audio_path": str(video_path / "final_audio.wav").replace('\\', '/'),
            "topic_index": topic_index, "music_prompt": music_prompt,
        }
        (video_path / "metadata.json").write_text(json.dumps(metadata, indent=4), encoding='utf-8')
        (video_path / "tts_script.txt").write_text(tts_script, encoding='utf-8')
        (video_path / "video_prompt.txt").write_text(video_prompt, encoding='utf-8')
        (video_path / "music_prompt.txt").write_text(music_prompt, encoding='utf-8')
        (video_path / "subtitles.vtt").write_text(vtt_content, encoding='utf-8')

    def _format_vtt_timestamp(self, seconds: float) -> str:
        total_seconds, milliseconds = int(seconds), int((seconds * 1000) % 1000)
        hours, minutes, seconds_part = total_seconds // 3600, (total_seconds % 3600) // 60, total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}.{milliseconds:03d}"

    def _create_vtt_content(self, tokens: list) -> str:
        vtt_blocks = ["WEBVTT\n"]
        style_settings = "line:70% position:50% align:middle"
        for token in tokens:
            if not token.text.strip() or token.start_ts is None or token.end_ts is None: continue
            start, end = self._format_vtt_timestamp(token.start_ts), self._format_vtt_timestamp(token.end_ts)
            vtt_blocks.append(f"{start} --> {end} {style_settings}\n{token.text.strip()}")
        return "\n\n".join(vtt_blocks)

    def _generate_tts(self, text: str, voice_name: str):
        try:
            audio_chunks, all_tokens, cumulative_duration_s = [], [], 0.0
            for result in self.tts_pipeline(text, voice=voice_name):
                audio_chunk_np, current_tokens = result.audio.cpu().numpy(), result.tokens
                if cumulative_duration_s > 0:
                    for token in current_tokens:
                        if token.start_ts is not None: token.start_ts += cumulative_duration_s
                        if token.end_ts is not None: token.end_ts += cumulative_duration_s
                all_tokens.extend(current_tokens)
                audio_chunks.append(audio_chunk_np)
                cumulative_duration_s += len(audio_chunk_np) / 24000
            if not audio_chunks: return None, None, 0
            return np.concatenate(audio_chunks), all_tokens, cumulative_duration_s
        except Exception as e:
            print(f"   - TTS Generation Error: {e}"); return None, None, 0

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
            for chunk in (c for c in raw_output.split('### ') if c):
                header, content = chunk.split('\n', 1)
                parts[header.strip()] = content.strip()
            username = parts.get("Username", "").split()[0].lower().strip() if parts.get("Username") else "user"
            return (parts.get("TTS Script"), parts.get("Short Description"), parts.get("Video Prompt"), username, parts.get("Music Prompt"))
        except (ValueError, KeyError) as e:
            print(f"   - ❌ LLM Parsing Error: {e}"); return None, None, None, None, None
            
    def _consolidate_subtitle_tokens(self, tokens: list) -> list:
        if not tokens: return []
        valid_raw_tokens = [t for t in tokens if t.text.strip() and t.start_ts is not None and t.end_ts is not None]
        if not valid_raw_tokens: return []
        consolidated = [valid_raw_tokens[0]]
        for token in valid_raw_tokens[1:]:
            prev = consolidated[-1]
            if token.text.strip() in PUNCTUATION_SET or (token.end_ts - token.start_ts) < MIN_SUBTITLE_DURATION_S:
                prev.text += " " + token.text.strip() if token.text.strip() not in PUNCTUATION_SET else token.text.strip()
                prev.end_ts = token.end_ts
            else:
                consolidated.append(token)
        return consolidated

    def run(self):
        self._initialize_systems()
        self._process_existing_watch_stats()
        print("\n" + "=" * 50)
        print("🤖 AI Content Factory (Intelligent Mixing, VTT, Auto-Download)")
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

def main():
    parser = argparse.ArgumentParser(description="AI Content Factory with intelligent audio mixing.")
    parser.add_argument("--output_dir", default="generated_content", help="Directory to save generated video assets.")
    parser.add_argument("--interval", type=int, default=10, help="Interval in seconds between generation cycles.")
    parser.add_argument("--model_name", default="Qwen/Qwen3-4B", help="Name of the Hugging Face model to use.")
    parser.add_argument("--acestep_checkpoint_path", default="./acestep_checkpoints", help="Path to ACEStep checkpoints. Will be downloaded here if not found.")
    parser.add_argument("--bf16", action='store_true', help="Use bfloat16 for faster music generation.", default=True)
    parser.add_argument("--cpu_offload", action='store_true', help="Enable CPU offloading for music generation to save VRAM.")
    args = parser.parse_args()
    factory = AIContentFactory(**vars(args))
    factory.run()

if __name__ == "__main__":
    main()