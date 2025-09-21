#!/bin/bash
# Simple deployment script for Koyeb

echo "🚀 Deploying SD 3.5 Medium test to Koyeb..."

# Create a simple koyeb.yaml config
cat > koyeb.yaml << EOF
services:
  - name: sd35-test
    type: worker
    buildpack: python
    run_command: python simple_test.py
    work_directory: /app
    instance_type: gpu-nvidia-rtx-4000-sff-ada
    regions:
      - eu
    env:
      - name: PYTHON_VERSION
        value: "3.11"
EOF

echo "✅ Created koyeb.yaml configuration"
echo "📋 Configuration:"
cat koyeb.yaml

echo ""
echo "🎯 Next steps:"
echo "1. Go to https://app.koyeb.com"
echo "2. Create new service"
echo "3. Use repository: https://github.com/aybanda/sd35-koyeb-test"
echo "4. Use these settings:"
echo "   - Buildpack: Python"
echo "   - Run Command: python simple_test.py"
echo "   - Instance: RTX-4000-SFF-ADA"
echo "   - Region: Europe"
echo "   - Type: Worker"
echo ""
echo "🏁 Ready to deploy!"
