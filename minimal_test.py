#!/usr/bin/env python3
"""
Minimal Stable Diffusion 3.5 Medium Test
GitHub Issue #1042: Add model: Stable Diffusion 3.5 medium (512x512)
"""

import sys
import time
from datetime import datetime

def main():
    print("🎯 Stable Diffusion 3.5 Medium Test")
    print("Issue #1042: Add model: Stable Diffusion 3.5 medium (512x512)")
    print("=" * 60)
    
    # Check Python
    print(f"🐍 Python: {sys.version}")
    
    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"🔥 CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not available")
        return 1
    
    # Check diffusers
    try:
        import diffusers
        print(f"✅ Diffusers: {diffusers.__version__}")
    except ImportError:
        print("❌ Diffusers not available")
        return 1
    
    # Test SD 3.5 Medium
    print("\n🚀 Testing Stable Diffusion 3.5 Medium...")
    
    try:
        from diffusers import StableDiffusionPipeline
        
        model_id = "stabilityai/stable-diffusion-3-medium"
        prompt = "a photo of an astronaut riding a horse on mars"
        
        print(f"📦 Loading: {model_id}")
        
        # Load model
        start = time.time()
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        )
        load_time = time.time() - start
        print(f"✅ Loaded in {load_time:.2f}s")
        
        # Move to device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        print(f"🖥️ Device: {device}")
        
        # Test inference
        print("\n🎬 Performance test...")
        
        # Warmup
        with torch.no_grad():
            _ = pipe(prompt, num_inference_steps=1, output_type="latent")
        
        # Test runs
        times = []
        for i in range(3):
            print(f"🔄 Run {i+1}/3...")
            start = time.time()
            with torch.no_grad():
                result = pipe(prompt, num_inference_steps=20, output_type="pil")
            end = time.time()
            
            inference_time = end - start
            times.append(inference_time)
            fps = 1.0 / inference_time
            print(f"   ⏱️ {inference_time:.2f}s, FPS: {fps:.3f}")
        
        # Results
        avg_time = sum(times) / len(times)
        avg_fps = 1.0 / avg_time
        
        print("\n" + "=" * 60)
        print("📊 RESULTS")
        print("=" * 60)
        print(f"🎯 Target FPS: 0.3")
        print(f"📈 Average FPS: {avg_fps:.3f}")
        print(f"📉 Average time: {avg_time:.2f}s")
        print(f"🖥️ Device: {device}")
        print(f"⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if avg_fps >= 0.3:
            print("✅ TARGET ACHIEVED!")
        else:
            improvement = 0.3 / avg_fps
            print(f"⚠️ Need {improvement:.1f}x improvement")
        
        print("\n📋 GITHUB ISSUE REPORT")
        print("-" * 40)
        print(f"**Performance Results for Issue #1042:**")
        print(f"- Average FPS: {avg_fps:.3f}")
        print(f"- Target FPS: 0.3")
        print(f"- Baseline FPS: 0.06")
        print(f"- Device: {device}")
        print(f"- Model: {model_id}")
        print(f"- Test date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n🏁 Test completed!")
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
