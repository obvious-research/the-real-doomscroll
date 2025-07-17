import os
import time
import threading
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import argparse

from ltx_video.inference import (
    infer,
    InferenceConfig,
    load_pipeline_config,
    create_ltx_video_pipeline,
    create_latent_upsampler,
    get_device,
    seed_everething,
    calculate_padding,
    get_unique_filename,
    SkipLayerStrategy,
    LTXMultiScalePipeline,
    load_media_file,
    prepare_conditioning,
)
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.autoencoders.latent_upsampler import LatentUpsampler
from transformers import T5EncoderModel, T5Tokenizer
from safetensors import safe_open
import json
import torch
from huggingface_hub import hf_hub_download


@dataclass
class MonitorConfig:
    prompt_file: str = field(
        default="prompts.txt",
        metadata={"help": "Path to the text file containing prompts to monitor"}
    )
    check_interval: float = field(
        default=1.0,
        metadata={"help": "Interval in seconds to check for new prompts"}
    )
    pipeline_config: str = field(
        #default="configs/ltxv-2b-0.9.8-distilled.yaml",
        default="configs/ltxv-13b-0.9.8-distilled.yaml",
        metadata={"help": "Path to the pipeline config file"},
    )
    seed: int = field(
        default=171198, metadata={"help": "Random seed for the inference"}
    )
    height: int = field(
        default=704, metadata={"help": "Height of the output video frames"}
    )
    width: int = field(
        default=550, metadata={"help": "Width of the output video frames"}
    )
    num_frames: int = field(
        default=240,
        metadata={"help": "Number of frames to generate in the output video"},
    )
    frame_rate: int = field(
        default=30, metadata={"help": "Frame rate for the output video"}
    )
    offload_to_cpu: bool = field(
        default=False, metadata={"help": "Offloading unnecessary computations to CPU."}
    )
    negative_prompt: str = field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        metadata={"help": "Negative prompt for undesired features"},
    )
    output_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the folder to save the output video"},
    )
    use_compile: bool = field(
        default=True,
        metadata={"help": "Use torch.compile to optimize models for faster inference"}
    )


class VideoGenerator:
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.pipeline = None
        self.pipeline_config = None
        self.device = None
        self.skip_layer_strategy = None
        self.output_dir = None
        self.processed_lines = set()
        self.generation_queue = []
        self.queue_lock = threading.Lock()
        self.is_running = False
        self.video_paths_file = "generated_videos.txt"
        
        # Initialize pipeline
        self._initialize_pipeline()
        
    def _compile_model(self, model, model_name: str):
        """Compile a model using torch.compile for faster inference."""
        if self.config.use_compile and hasattr(torch, 'compile'):
            try:
                print(f"⚡ Compiling {model_name} for faster inference...")
                # Use a more conservative compilation mode to avoid type issues
                compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
                print(f"✅ {model_name} compiled successfully!")
                return compiled_model
            except Exception as e:
                print(f"⚠️  Failed to compile {model_name}: {e}")
                print(f"   Continuing with uncompiled {model_name}")
                return model
        else:
            if not hasattr(torch, 'compile'):
                print(f"⚠️  torch.compile not available (PyTorch version too old)")
            return model
        
    def _safe_compile_pipeline(self):
        """Safely compile pipeline components with fallback options."""
        if not self.config.use_compile:
            print("⚡ Torch compilation disabled")
            return
            
        print("⚡ Attempting to compile pipeline components...")
        
        try:
            # Only compile the transformer and patchifier, skip VAE and text_encoder
            # to avoid type assertion issues
            if hasattr(self.pipeline, 'transformer'):
                self.pipeline.transformer = self._compile_model(self.pipeline.transformer, "transformer")
            
            if hasattr(self.pipeline, 'patchifier'):
                self.pipeline.patchifier = self._compile_model(self.pipeline.patchifier, "patchifier")
            
            # Skip VAE and text_encoder compilation to avoid type assertion issues
            print("⚠️  Skipping VAE and text_encoder compilation to avoid type assertion issues")
            print("   This is necessary because the pipeline performs type checks on these components")
            
        except Exception as e:
            print(f"❌ Pipeline compilation failed: {e}")
            print("   Continuing with uncompiled pipeline")
            print("   If you continue to have issues, try running with --no_compile")
        
    def _initialize_pipeline(self):
        """Initialize the pipeline and related components."""
        print("🚀 Initializing LTX-Video pipeline...")
        print("=" * 60)
        
        # Load pipeline config
        self.pipeline_config = load_pipeline_config(self.config.pipeline_config)
        print(f"📋 Loaded pipeline config: {self.config.pipeline_config}")
        
        # Setup device and output directory
        self.device = get_device()
        print(f"💻 Using device: {self.device}")
        
        # Handle output path - use default if None
        if self.config.output_path is None:
            self.output_dir = Path(f"outputs_new4/{datetime.today().strftime('%Y-%m-%d')}")
        else:
            self.output_dir = Path(self.config.output_path)
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {self.output_dir}")
        
        # Download models if needed
        print("📥 Checking for model files...")
        ltxv_model_name_or_path = self.pipeline_config["checkpoint_path"]
        if not os.path.isfile(ltxv_model_name_or_path):
            print(f"⬇️  Downloading model: {ltxv_model_name_or_path}")
            ltxv_model_path = hf_hub_download(
                repo_id="Lightricks/LTX-Video",
                filename=ltxv_model_name_or_path,
                repo_type="model",
            )
        else:
            ltxv_model_path = ltxv_model_name_or_path
            print(f"✅ Model found locally: {ltxv_model_path}")
            
        spatial_upscaler_model_name_or_path = self.pipeline_config.get(
            "spatial_upscaler_model_path"
        )
        if spatial_upscaler_model_name_or_path and not os.path.isfile(
            spatial_upscaler_model_name_or_path
        ):
            print(f"⬇️  Downloading spatial upscaler: {spatial_upscaler_model_name_or_path}")
            spatial_upscaler_model_path = hf_hub_download(
                repo_id="Lightricks/LTX-Video",
                filename=spatial_upscaler_model_name_or_path,
                repo_type="model",
            )
        else:
            spatial_upscaler_model_path = spatial_upscaler_model_name_or_path
            
        # Create pipeline
        print("🔧 Creating video generation pipeline...")
        precision = self.pipeline_config["precision"]
        text_encoder_model_name_or_path = self.pipeline_config["text_encoder_model_name_or_path"]
        sampler = self.pipeline_config.get("sampler", None)
        prompt_enhancer_image_caption_model_name_or_path = self.pipeline_config[
            "prompt_enhancer_image_caption_model_name_or_path"
        ]
        prompt_enhancer_llm_model_name_or_path = self.pipeline_config[
            "prompt_enhancer_llm_model_name_or_path"
        ]
        
        # Check if prompt enhancement is needed
        prompt_enhancement_words_threshold = self.pipeline_config[
            "prompt_enhancement_words_threshold"
        ]
        enhance_prompt = prompt_enhancement_words_threshold > 0
        
        self.pipeline = create_ltx_video_pipeline(
            ckpt_path=ltxv_model_path,
            precision=precision,
            text_encoder_model_name_or_path=text_encoder_model_name_or_path,
            sampler=sampler,
            device=self.device,
            enhance_prompt=enhance_prompt,
            prompt_enhancer_image_caption_model_name_or_path=prompt_enhancer_image_caption_model_name_or_path,
            prompt_enhancer_llm_model_name_or_path=prompt_enhancer_llm_model_name_or_path,
        )
        
        # Safely compile pipeline components
        self._safe_compile_pipeline()
        
        # Setup multi-scale pipeline if needed
        if self.pipeline_config.get("pipeline_type", None) == "multi-scale":
            if not spatial_upscaler_model_path:
                raise ValueError(
                    "spatial upscaler model path is missing from pipeline config file and is required for multi-scale rendering"
                )
            print("🔧 Setting up multi-scale pipeline...")
            latent_upsampler = create_latent_upsampler(
                spatial_upscaler_model_path, self.pipeline.device
            )
            
            # Compile the latent upsampler if using compile
            if self.config.use_compile:
                latent_upsampler = self._compile_model(latent_upsampler, "latent_upsampler")
                
            self.pipeline = LTXMultiScalePipeline(
                self.pipeline, latent_upsampler=latent_upsampler)
                
        # Setup skip layer strategy
        stg_mode = self.pipeline_config.get("stg_mode", "attention_values")
        if stg_mode.lower() == "stg_av" or stg_mode.lower() == "attention_values":
            self.skip_layer_strategy = SkipLayerStrategy.AttentionValues
        elif stg_mode.lower() == "stg_as" or stg_mode.lower() == "attention_skip":
            self.skip_layer_strategy = SkipLayerStrategy.AttentionSkip
        elif stg_mode.lower() == "stg_r" or stg_mode.lower() == "residual":
            self.skip_layer_strategy = SkipLayerStrategy.Residual
        elif stg_mode.lower() == "stg_t" or stg_mode.lower() == "transformer_block":
            self.skip_layer_strategy = SkipLayerStrategy.TransformerBlock
        else:
            raise ValueError(f"Invalid spatiotemporal guidance mode: {stg_mode}")
            
        print("✅ Pipeline initialized successfully!")
        print("=" * 60)
        
        # Validate pipeline components
        self._validate_pipeline()
        
    def _validate_pipeline(self):
        """Validate that the pipeline components are working correctly."""
        print("🔍 Validating pipeline components...")
        
        try:
            # Check if pipeline has required attributes
            required_attrs = ['transformer', 'vae', 'text_encoder', 'tokenizer', 'scheduler']
            missing_attrs = []
            
            for attr in required_attrs:
                if not hasattr(self.pipeline, attr):
                    missing_attrs.append(attr)
            
            if missing_attrs:
                print(f"⚠️  Missing pipeline attributes: {missing_attrs}")
            else:
                print("✅ All required pipeline components found")
            
            # Check device placement
            if hasattr(self.pipeline, 'transformer'):
                transformer_device = next(self.pipeline.transformer.parameters()).device
                print(f"📱 Transformer device: {transformer_device}")
            
            if hasattr(self.pipeline, 'vae'):
                vae_device = next(self.pipeline.vae.parameters()).device
                print(f"📱 VAE device: {vae_device}")
            
            if hasattr(self.pipeline, 'text_encoder'):
                text_encoder_device = next(self.pipeline.text_encoder.parameters()).device
                print(f"📱 Text encoder device: {text_encoder_device}")
            
            # Check pipeline config
            print(f"📋 Pipeline config keys: {list(self.pipeline_config.keys())}")
            print(f"🎯 Skip layer strategy: {self.skip_layer_strategy}")
            
            # Test basic forward pass with minimal input
            print("🧪 Testing basic forward pass...")
            try:
                with torch.no_grad():
                    # Create minimal test input
                    test_height = 32
                    test_width = 32
                    test_frames = 9
                    
                    test_sample = {
                        "prompt": "test",
                        "prompt_attention_mask": None,
                        "negative_prompt": "test",
                        "negative_prompt_attention_mask": None,
                    }
                    
                    test_generator = torch.Generator(device=self.device).manual_seed(42)
                    
                    # Try a minimal forward pass
                    test_images = self.pipeline(
                        **self.pipeline_config,
                        skip_layer_strategy=self.skip_layer_strategy,
                        generator=test_generator,
                        output_type="pt",
                        callback_on_step_end=None,
                        height=test_height,
                        width=test_width,
                        num_frames=test_frames,
                        frame_rate=30,
                        **test_sample,
                        media_items=None,
                        conditioning_items=None,
                        is_video=True,
                        vae_per_channel_normalize=True,
                        image_cond_noise_scale=0.15,
                        mixed_precision=(self.pipeline_config["precision"] == "mixed_precision"),
                        offload_to_cpu=False,
                        device=self.device,
                        enhance_prompt=False,
                    ).images
                    
                    print(f"✅ Forward pass test successful! Output shape: {test_images.shape}")
                    
            except Exception as test_error:
                print(f"❌ Forward pass test failed:")
                print(f"   Error type: {type(test_error).__name__}")
                print(f"   Error message: {str(test_error)}")
                print("   This might indicate an issue with the pipeline configuration")
            
            print("✅ Pipeline validation completed")
            
        except Exception as e:
            print(f"⚠️  Pipeline validation failed: {e}")
            print("   Continuing anyway...")
        
    def _save_video_path(self, video_path: str, prompt: str, seed: int, generation_time: float):
        """Save video path to a separate file with metadata."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.video_paths_file, "a") as f:
            f.write(f"{timestamp} | {video_path} | Prompt: '{prompt}' | Seed: {seed} | Time: {generation_time:.2f}s\n")
        
    def _generate_video(self, prompt: str, seed: int) -> str:
        """Generate a video for the given prompt."""
        start_time = time.time()
        detection_time = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n🎬 Starting video generation at {detection_time}")
        print(f"📝 Prompt: '{prompt}'")
        print(f"🎲 Seed: {seed}")
        print(f"📐 Resolution: {self.config.width}x{self.config.height}")
        print(f"🎞️  Frames: {self.config.num_frames} @ {self.config.frame_rate}fps")
        print(f"⚡ Using compiled models: {self.config.use_compile}")
        print("-" * 60)
        
        try:
            # Set seed
            seed_everething(seed)
            
            # Calculate padding and dimensions
            height_padded = ((self.config.height - 1) // 32 + 1) * 32
            width_padded = ((self.config.width - 1) // 32 + 1) * 32
            num_frames_padded = ((self.config.num_frames - 2) // 8 + 1) * 8 + 1
            
            print(f"🔧 Calculated padded dimensions: {height_padded}x{width_padded}x{num_frames_padded}")
            
            padding = calculate_padding(
                self.config.height, self.config.width, height_padded, width_padded
            )
            print(f"📏 Padding: {padding}")
            
            # Prepare input for the pipeline
            sample = {
                "prompt": prompt,
                "prompt_attention_mask": None,
                "negative_prompt": self.config.negative_prompt,
                "negative_prompt_attention_mask": None,
            }
            
            print(f"📝 Sample input prepared: {list(sample.keys())}")
            
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Check offload settings
            if self.config.offload_to_cpu and not torch.cuda.is_available():
                offload_to_cpu = False
            else:
                offload_to_cpu = self.config.offload_to_cpu and self._get_total_gpu_memory() < 30
                
            print(f"💾 Offload to CPU: {offload_to_cpu}")
            print("🎨 Generating video frames...")
            
            # Generate video
            try:
                images = self.pipeline(
                    **self.pipeline_config,
                    skip_layer_strategy=self.skip_layer_strategy,
                    generator=generator,
                    output_type="pt",
                    callback_on_step_end=None,
                    height=height_padded,
                    width=width_padded,
                    num_frames=num_frames_padded,
                    frame_rate=self.config.frame_rate,
                    **sample,
                    media_items=None,
                    conditioning_items=None,
                    is_video=True,
                    vae_per_channel_normalize=True,
                    image_cond_noise_scale=0.15,
                    mixed_precision=(self.pipeline_config["precision"] == "mixed_precision"),
                    offload_to_cpu=offload_to_cpu,
                    device=self.device,
                    enhance_prompt=False,  # We'll handle this separately if needed
                ).images
                
                print(f"✅ Pipeline execution completed. Output shape: {images.shape}")
                
            except Exception as pipeline_error:
                print(f"❌ Pipeline execution failed:")
                print(f"   Error type: {type(pipeline_error).__name__}")
                print(f"   Error message: {str(pipeline_error)}")
                print(f"   Pipeline config keys: {list(self.pipeline_config.keys())}")
                print(f"   Skip layer strategy: {self.skip_layer_strategy}")
                raise pipeline_error
            
            # Crop the padded images to the desired resolution and number of frames
            (pad_left, pad_right, pad_top, pad_bottom) = padding
            pad_bottom = -pad_bottom
            pad_right = -pad_right
            if pad_bottom == 0:
                pad_bottom = images.shape[3]
            if pad_right == 0:
                pad_right = images.shape[4]
            images = images[:, :, : self.config.num_frames,
                            pad_top:pad_bottom, pad_left:pad_right]
            
            print(f"✂️  Cropped images shape: {images.shape}")
            print("💾 Saving video...")
            
            # Save video
            import imageio
            import numpy as np
            
            for i in range(images.shape[0]):
                # Gathering from B, C, F, H, W to C, F, H, W and then permute to F, H, W, C
                video_np = images[i].permute(1, 2, 3, 0).cpu().float().numpy()
                # Unnormalizing images to [0, 255] range
                video_np = (video_np * 255).astype(np.uint8)
                fps = self.config.frame_rate
                height, width = video_np.shape[1:3]
                
                print(f"📹 Video tensor shape: {video_np.shape}, FPS: {fps}")
                
                # In case a single image is generated
                if video_np.shape[0] == 1:
                    output_filename = get_unique_filename(
                        f"image_output_{i}",
                        ".png",
                        prompt=prompt,
                        seed=seed,
                        resolution=(height, width, self.config.num_frames),
                        dir=self.output_dir,
                    )
                    imageio.imwrite(output_filename, video_np[0])
                else:
                    output_filename = get_unique_filename(
                        f"video_output_{i}",
                        ".mp4",
                        prompt=prompt,
                        seed=seed,
                        resolution=(height, width, self.config.num_frames),
                        dir=self.output_dir,
                    )
                    
                    # Write video
                    with imageio.get_writer(output_filename, fps=fps) as video:
                        for frame in video_np:
                            video.append_data(frame)
                
                generation_time = time.time() - start_time
                completion_time = datetime.now().strftime("%H:%M:%S")
                
                print(f"✅ Video saved successfully!")
                print(f"📁 Path: {output_filename}")
                print(f"⏱️  Generation time: {generation_time:.2f} seconds")
                print(f"🕐 Completed at: {completion_time}")
                print("-" * 60)
                
                # Save video path to file
                self._save_video_path(str(output_filename), prompt, seed, generation_time)
                
                return str(output_filename)
                
        except Exception as e:
            print(f"❌ Error generating video for prompt '{prompt}':")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            
            # Print additional debugging information
            import traceback
            print(f"   Full traceback:")
            traceback.print_exc()
            
            # Check GPU memory if using CUDA
            if torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    print(f"   GPU memory used: {gpu_memory:.2f}GB / {gpu_memory_total:.2f}GB")
                except Exception as mem_error:
                    print(f"   Could not get GPU memory info: {mem_error}")
            
            raise e
            
    def _get_total_gpu_memory(self):
        """Get total GPU memory in GB."""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return total_memory
        return 0
        
    def _process_prompt_file(self):
        """Read the prompt file and add new prompts to the queue."""
        if not os.path.exists(self.config.prompt_file):
            return
            
        with open(self.config.prompt_file, 'r') as f:
            lines = f.readlines()
            
        with self.queue_lock:
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if line and line not in self.processed_lines and not line.startswith('#'):
                    # Parse prompt and optional seed
                    parts = line.split('|')
                    prompt = parts[0].strip()
                    seed = int(parts[1].strip()) if len(parts) > 1 else self.config.seed
                    
                    self.generation_queue.append((prompt, seed))
                    self.processed_lines.add(line)
                    
                    current_time = datetime.now().strftime("%H:%M:%S")
                    print(f"🆕 [{current_time}] New prompt detected!")
                    print(f"📝 Added to queue: '{prompt}' with seed {seed}")
                    print(f"📊 Queue length: {len(self.generation_queue)}")
                    
    def _process_queue(self):
        """Process the generation queue."""
        with self.queue_lock:
            if not self.generation_queue:
                return
                
            prompt, seed = self.generation_queue.pop(0)
            
        try:
            self._generate_video(prompt, seed)
        except Exception as e:
            print(f"❌ Error generating video for prompt '{prompt}': {e}")
            
    def run(self):
        """Main monitoring loop."""
        print("🎥 LTX-Video Generation Monitor")
        print("=" * 60)
        print(f"📂 Monitoring file: {self.config.prompt_file}")
        print(f"⏰ Check interval: {self.config.check_interval} seconds")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📄 Video paths will be saved to: {self.video_paths_file}")
        print(f"⚡ Torch compile enabled: {self.config.use_compile}")
        print("🛑 Press Ctrl+C to stop")
        print("=" * 60)
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Check for new prompts
                self._process_prompt_file()
                
                # Process queue
                self._process_queue()
                
                # Wait before next check
                time.sleep(self.config.check_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping monitor...")
            print("👋 Goodbye!")
            self.is_running = False


def main():
    parser = argparse.ArgumentParser(description="Monitor a text file for prompts and generate videos")
    parser.add_argument("--prompt_file", default="prompts.txt", help="Path to the text file containing prompts")
    parser.add_argument("--check_interval", type=float, default=1.0, help="Interval in seconds to check for new prompts")
    #parser.add_argument("--pipeline_config", default="configs/ltxv-2b-0.9.8-distilled.yaml", help="Path to the pipeline config file")
    parser.add_argument("--pipeline_config", default="configs/ltxv-13b-0.9.8-distilled.yaml", help="Path to the pipeline config file")
    parser.add_argument("--seed", type=int, default=171198, help="Default random seed for inference")
    parser.add_argument("--height", type=int, default=704, help="Height of the output video frames")
    parser.add_argument("--width", type=int, default=550, help="Width of the output video frames")
    parser.add_argument("--num_frames", type=int, default=240, help="Number of frames to generate")
    parser.add_argument("--frame_rate", type=int, default=30, help="Frame rate for the output video")
    parser.add_argument("--output_path", default=None, help="Path to save output videos")
    parser.add_argument("--negative_prompt", default="worst quality, inconsistent motion, blurry, jittery, distorted", help="Negative prompt")
    parser.add_argument("--offload_to_cpu", action="store_true", help="Offload to CPU")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile optimization")
    
    args = parser.parse_args()
    
    # Create config
    config = MonitorConfig(
        prompt_file=args.prompt_file,
        check_interval=args.check_interval,
        pipeline_config=args.pipeline_config,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        output_path=args.output_path,
        negative_prompt=args.negative_prompt,
        offload_to_cpu=args.offload_to_cpu,
        use_compile=not args.no_compile,
    )
    
    # Create and run generator
    generator = VideoGenerator(config)
    generator.run()


if __name__ == "__main__":
    main() 