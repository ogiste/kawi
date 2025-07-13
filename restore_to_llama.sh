#!/bin/bash

echo "🔄 Restoring CSM to use Llama 3.2..."
echo "🔐 Requires HuggingFace authorization for Llama models"

# Test the restoration
echo "🧪 Running restoration test..."
python3 test_llama_restoration.py

if [ $? -eq 0 ]; then
    echo "✅ Restoration successful!"
    echo "🎉 You can now use:"
    echo "  python test_streaming.py --text 'Hello from Llama 3.2!'"
    echo "  python deploy_vast.py --all"
    echo ""
    echo "🔐 Benefits of Llama 3.2 restoration:"
    echo "  • Native tokenizer compatibility"
    echo "  • Perfect vocab size match (128,256)"
    echo "  • Original CSM training alignment"
    echo "  • Optimal performance and quality"
    echo "  • No token clipping or mapping needed"
    echo ""
    echo "⚠️  Note: Requires HuggingFace authorization for meta-llama models"
    echo "  Run: huggingface-cli login"
else
    echo "❌ Restoration failed. Check the logs above."
    echo "💡 Make sure you have access to meta-llama/Llama-3.2-1B"
    echo "  and run: huggingface-cli login"
fi 