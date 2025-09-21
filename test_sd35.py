#!/usr/bin/env python3
"""
Simple Stable Diffusion 3.5 Medium Performance Test
GitHub Issue #1042: Add model: Stable Diffusion 3.5 medium (512x512)

Target: 0.3 FPS on batch 1
Baseline: 0.06 FPS on batch 1
"""

import time
import torch
import psutil
import os
import sys
from datetime import datetime

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def check_environment():
    """Check the testing environment"""
    print("🔍 Environment Check")
    print("-" * 40)
    
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print(f"🔥 CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"🔥 CUDA version: {torch.version.cuda}")
        print(f"🔥 GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"🔥 GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Check for Tenstorrent hardware
    tt_devices = []
    for i in range(10):
        device_path = f"/dev/tenstorrent{i}"
        if os.path.exists(device_path):
            tt_devices.append(i)
    
    if tt_devices:
        print(f"🧠 Tenstorrent devices found: {tt_devices}")
    else:
        print("⚠️ No Tenstorrent devices found")
    
    # Check SFPI
    sfpi_path = "/opt/tenstorrent/sfpi"
    if os.path.exists(sfpi_path):
        print(f"✅ SFPI found at {sfpi_path}")
    else:
        print(f"⚠️ SFPI not found at {sfpi_path}")
    
    # Check TTNN availability
    try:
        import ttnn
        print("✅ ttnn available")
        return True, tt_devices
    except ImportError:
        print("❌ ttnn not available")
        return False, tt_devices

def test_stable_diffusion_35_medium():
    """Test Stable Diffusion 3.5 Medium model performance"""
    print("\n🚀 Stable Diffusion 3.5 Medium Performance Test")
    print("=" * 60)
    
    # Model configuration
    model_id = "stabilityai/stable-diffusion-3-medium"
    prompt = "a photo of an astronaut riding a horse on mars"
    
    print(f"📦 Model: {model_id}")
    print(f"🎯 Target FPS: 0.3")
    print(f"📝 Prompt: {prompt}")
    print(f"⏰ Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Memory before loading
    mem_before = get_memory_usage()
    print(f"💾 Memory before loading: {mem_before:.2f} MB")
    
    # Load model
    print("📥 Loading model...")
    load_start = time.time()
    try:
        from diffusers import StableDiffusionPipeline
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        )
        load_time = time.time() - load_start
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Memory after loading
    mem_after_load = get_memory_usage()
    print(f"💾 Memory after loading: {mem_after_load:.2f} MB")
    print(f"💾 Memory used by model: {mem_after_load - mem_before:.2f} MB")
    
    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")
    
    if device == "cuda":
        pipe = pipe.to(device)
        print("✅ Model moved to CUDA")
    
    # Test inference
    print("\n🎬 Running inference tests...")
    
    # Warmup runs
    print("🔥 Warming up (3 runs)...")
    for i in range(3):
        with torch.no_grad():
            _ = pipe(prompt, num_inference_steps=1, output_type="latent")
        print(f"   Warmup {i+1}/3 completed")
    
    # Actual test runs
    num_runs = 5
    inference_times = []
    
    print(f"\n🔄 Running {num_runs} performance tests...")
    for i in range(num_runs):
        print(f"🔄 Run {i+1}/{num_runs}...")
        
        start_time = time.time()
        with torch.no_grad():
            result = pipe(
                prompt,
                num_inference_steps=20,
                output_type="pil",
                guidance_scale=7.5,
                height=512,
                width=512
            )
        end_time = time.time()
        
        inference_time = end_time - start_time
        inference_times.append(inference_time)
        fps = 1.0 / inference_time
        
        print(f"   ⏱️ Inference time: {inference_time:.2f} seconds")
        print(f"   🎯 FPS: {fps:.3f}")
        
        # Save first image
        if i == 0 and result.images:
            image = result.images[0]
            image.save(f"sd35_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            print(f"   💾 Saved output image")
    
    # Calculate statistics
    avg_inference_time = sum(inference_times) / len(inference_times)
    min_inference_time = min(inference_times)
    max_inference_time = max(inference_times)
    avg_fps = 1.0 / avg_inference_time
    max_fps = 1.0 / min_inference_time
    
    # Memory after inference
    mem_after_inference = get_memory_usage()
    
    # Results
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE RESULTS")
    print("=" * 60)
    print(f"🎯 Target FPS: 0.3")
    print(f"📈 Average FPS: {avg_fps:.3f}")
    print(f"📈 Best FPS: {max_fps:.3f}")
    print(f"📉 Average inference time: {avg_inference_time:.2f} seconds")
    print(f"📉 Best inference time: {min_inference_time:.2f} seconds")
    print(f"📉 Worst inference time: {max_inference_time:.2f} seconds")
    print(f"💾 Total memory used: {mem_after_inference - mem_before:.2f} MB")
    print(f"⏰ Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Performance analysis
    print("\n🎯 PERFORMANCE ANALYSIS")
    print("-" * 40)
    if avg_fps >= 0.3:
        print("✅ TARGET ACHIEVED! Performance meets or exceeds 0.3 FPS")
        improvement = avg_fps / 0.06
        print(f"🚀 Performance improvement: {improvement:.1f}x over baseline (0.06 FPS)")
    else:
        improvement_needed = 0.3 / avg_fps
        current_improvement = avg_fps / 0.06
        print(f"⚠️ TARGET NOT MET. Need {improvement_needed:.1f}x improvement to reach 0.3 FPS")
        print(f"📊 Current improvement over baseline: {current_improvement:.1f}x")
    
    # Report for GitHub issue
    print("\n📋 GITHUB ISSUE REPORT")
    print("-" * 40)
    print(f"**Performance Results for Issue #1042:**")
    print(f"- Average FPS: {avg_fps:.3f}")
    print(f"- Best FPS: {max_fps:.3f}")
    print(f"- Target FPS: 0.3")
    print(f"- Baseline FPS: 0.06")
    print(f"- Device: {device}")
    print(f"- Model: {model_id}")
    print(f"- Memory used: {mem_after_inference - mem_before:.2f} MB")
    print(f"- Test date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        'avg_fps': avg_fps,
        'max_fps': max_fps,
        'avg_time': avg_inference_time,
        'device': device,
        'memory_used': mem_after_inference - mem_before,
        'target_met': avg_fps >= 0.3
    }

if __name__ == "__main__":
    print("🎯 Stable Diffusion 3.5 Medium Test Suite")
    print("Issue #1042: Add model: Stable Diffusion 3.5 medium (512x512)")
    print("=" * 60)
    
    # Check environment
    ttnn_available, tt_devices = check_environment()
    
    # Test basic model functionality
    results = test_stable_diffusion_35_medium()
    
    if results:
        # Final summary
        print("\n🏁 FINAL SUMMARY")
        print("=" * 60)
        print(f"✅ Test completed successfully")
        print(f"📊 Average FPS: {results['avg_fps']:.3f}")
        print(f"🎯 Target met: {'Yes' if results['target_met'] else 'No'}")
        print(f"🖥️ Device used: {results['device']}")
        print(f"🧠 TTNN available: {'Yes' if ttnn_available else 'No'}")
        if tt_devices:
            print(f"🧠 Tenstorrent devices: {tt_devices}")
        
        print("\n📋 Next steps:")
        print("1. Report these results to GitHub issue #1042")
        print("2. If target not met, investigate TTNN optimization")
        print("3. Consider batch size optimization")
        print("4. Test with different model configurations")
    
    print("\n🏁 Test completed!")
    print("📋 Report these results back to GitHub issue #1042")
