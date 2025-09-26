#!/usr/bin/env python3
"""
Simple Stable Diffusion 3.5 Medium Performance Test
GitHub Issue #1042: Add model: Stable Diffusion 3.5 medium (512x512)

Target: 0.3 FPS on batch 1
Baseline: 0.06 FPS on batch 1
"""

import time
import sys
import os
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
        elif self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Model loading in progress...')
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress default logging to reduce noise
        pass

def start_health_server():
    """Start a simple health check server on port 8000"""
    server = HTTPServer(('0.0.0.0', 8000), HealthCheckHandler)
    server.serve_forever()

def main():
    # Start health check server in background
    health_thread = threading.Thread(target=start_health_server, daemon=True)
    health_thread.start()
    print("🏥 Health check server started on port 8000")
    
    print("🎯 Stable Diffusion 3.5 Medium Performance Test Suite")
    print("Issue #1042: Add model: Stable Diffusion 3.5 medium (512x512)")
    print("Testing actual SD 3.5 Medium model")
    print("=" * 60)
    
    # Check Python version
    print(f"🐍 Python: {sys.version}")
    
    # Check if we can import required packages
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"🔥 CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"🔥 CUDA version: {torch.version.cuda}")
            print(f"🔥 GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"🔥 GPU {i}: {torch.cuda.get_device_name(i)}")
        
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return 1
    
    try:
        import diffusers
        print(f"✅ Diffusers: {diffusers.__version__}")
    except ImportError as e:
        print(f"❌ Diffusers import failed: {e}")
        return 1
    
    try:
        import psutil
        print(f"✅ psutil available")
    except ImportError as e:
        print(f"❌ psutil import failed: {e}")
        return 1
    
    # Check HuggingFace authentication
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        print(f"✅ HuggingFace token available")
    else:
        print(f"⚠️ No HuggingFace token found - may need authentication for gated models")
    
    # Test Stable Diffusion 3.5 Medium
    print("\n🚀 Testing Stable Diffusion 3.5 Medium...")
    
    try:
        from diffusers import StableDiffusion3Pipeline
        
        # Use actual SD 3.5 Medium model
        model_id = "stabilityai/stable-diffusion-3.5-medium"
        prompt = "a photo of an astronaut riding a horse on mars"
        
        print(f"📦 Loading model: {model_id}")
        print(f"📝 Prompt: {prompt}")
        
        # Load model with HuggingFace authentication
        start_time = time.time()
        print("⏳ Loading model (this may take several minutes)...")
        
        # Start a background thread to keep logging during model loading
        def keep_alive():
            while True:
                time.sleep(60)  # Log every minute
                elapsed = time.time() - start_time
                print(f"⏳ Still loading model... ({elapsed:.0f}s elapsed)")
        
        keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
        keep_alive_thread.start()
        
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,  # SD 3.5 uses bfloat16
            token=hf_token  # Use HuggingFace token for gated models
        )
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        
        # Move to device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        print(f"🖥️ Using device: {device}")
        
        # Test inference
        print("\n🎬 Running performance test...")
        
        # Warmup
        print("🔥 Warming up...")
        with torch.no_grad():
            _ = pipe(prompt, num_inference_steps=1, output_type="latent")
        
        # Performance test
        num_runs = 3
        times = []
        
        for i in range(num_runs):
            print(f"🔄 Run {i+1}/{num_runs}...")
            start = time.time()
            with torch.no_grad():
                result = pipe(prompt, num_inference_steps=20, output_type="pil")
            end = time.time()
            
            inference_time = end - start
            times.append(inference_time)
            fps = 1.0 / inference_time
            print(f"   ⏱️ Time: {inference_time:.2f}s, FPS: {fps:.3f}")
        
        # Calculate average
        avg_time = sum(times) / len(times)
        avg_fps = 1.0 / avg_time
        
        # Results
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE RESULTS")
        print("=" * 60)
        print(f"🎯 Target FPS: 0.3")
        print(f"📈 Average FPS: {avg_fps:.3f}")
        print(f"📉 Average time: {avg_time:.2f} seconds")
        print(f"🖥️ Device: {device}")
        print(f"⏰ Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Performance analysis
        if avg_fps >= 0.3:
            print("✅ TARGET ACHIEVED! Performance meets or exceeds 0.3 FPS")
        else:
            improvement_needed = 0.3 / avg_fps
            print(f"⚠️ TARGET NOT MET. Need {improvement_needed:.1f}x improvement")
        
        # Report for GitHub issue
        print("\n📋 GITHUB ISSUE REPORT")
        print("-" * 40)
        print(f"**Performance Results for Issue #1042:**")
        print(f"- Average FPS: {avg_fps:.3f}")
        print(f"- Target FPS: 0.3")
        print(f"- Baseline FPS: 0.06")
        print(f"- Device: {device}")
        print(f"- Model: {model_id}")
        print(f"- Test date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n🏁 Test completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
