#!/usr/bin/env python3
"""
Flask web app for Stable Diffusion 3.5 Medium Performance Test
GitHub Issue #1042: Add model: Stable Diffusion 3.5 medium (512x512)
"""

import time
import sys
import os
from datetime import datetime
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SD 3.5 Medium Performance Test</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; color: #333; margin-bottom: 30px; }
        .status { padding: 15px; margin: 10px 0; border-radius: 5px; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        .results { background: #e2e3e5; color: #383d41; border: 1px solid #d6d8db; }
        .button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .button:hover { background: #0056b3; }
        pre { background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Stable Diffusion 3.5 Medium Performance Test</h1>
            <p>GitHub Issue #1042: Add model: Stable Diffusion 3.5 medium (512x512)</p>
            <p><strong>Target:</strong> 0.3 FPS on batch 1 | <strong>Baseline:</strong> 0.06 FPS on batch 1</p>
        </div>
        
        <div class="status info">
            <h3>📊 Test Status</h3>
            <p>Click the button below to run the performance test.</p>
        </div>
        
        <div style="text-align: center; margin: 30px 0;">
            <button class="button" onclick="runTest()">🚀 Run Performance Test</button>
        </div>
        
        <div id="results" style="display: none;">
            <div class="status results">
                <h3>📈 Test Results</h3>
                <pre id="output"></pre>
            </div>
        </div>
    </div>
    
    <script>
        function runTest() {
            document.getElementById('results').style.display = 'block';
            document.getElementById('output').textContent = 'Running test... Please wait...';
            
            fetch('/run_test')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('output').textContent = data.output;
                })
                .catch(error => {
                    document.getElementById('output').textContent = 'Error: ' + error.message;
                });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/run_test')
def run_test():
    """Run the Stable Diffusion 3.5 Medium performance test"""
    try:
        # Capture output
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        output = io.StringIO()
        
        with redirect_stdout(output), redirect_stderr(output):
            result = run_sd35_test()
        
        return jsonify({
            'success': True,
            'output': output.getvalue(),
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'output': f'Error: {str(e)}',
            'result': None
        })

def run_sd35_test():
    """Run the actual SD 3.5 Medium test"""
    print("🎯 Stable Diffusion 3.5 Medium Test Suite")
    print("Issue #1042: Add model: Stable Diffusion 3.5 medium (512x512)")
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
        return {'error': 'PyTorch import failed'}
    
    try:
        import diffusers
        print(f"✅ Diffusers: {diffusers.__version__}")
    except ImportError as e:
        print(f"❌ Diffusers import failed: {e}")
        return {'error': 'Diffusers import failed'}
    
    # Test Stable Diffusion
    print("\n🚀 Testing Stable Diffusion 3.5 Medium...")
    
    try:
        from diffusers import StableDiffusionPipeline
        
        model_id = "stabilityai/stable-diffusion-3-medium"
        prompt = "a photo of an astronaut riding a horse on mars"
        
        print(f"📦 Loading model: {model_id}")
        print(f"📝 Prompt: {prompt}")
        
        # Load model
        start_time = time.time()
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
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
        
        return {
            'success': True,
            'avg_fps': avg_fps,
            'avg_time': avg_time,
            'device': device,
            'target_met': avg_fps >= 0.3
        }
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
