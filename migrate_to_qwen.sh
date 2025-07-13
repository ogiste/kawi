#!/bin/bash

echo "🔄 Migrating CSM from Llama to Qwen2.5..."
echo "🔓 No authorization required - using fully open models!"

# Test the migration
echo "🧪 Running migration test..."
python3 test_qwen_migration.py

if [ $? -eq 0 ]; then
    echo "✅ Migration successful!"
    echo "🎉 You can now use:"
    echo "  python test_streaming.py --text 'Hello from Qwen2.5!'"
    echo "  python deploy_vast.py --all"
    echo ""
    echo "🔓 Benefits of Qwen2.5 migration:"
    echo "  • No HuggingFace authorization required"
    echo "  • Apache 2.0 license - fully open"
    echo "  • Better multilingual support (29+ languages)"
    echo "  • Improved performance and efficiency"
    echo "  • Easier deployment on vast.ai"
else
    echo "❌ Migration failed. Check the logs above."
fi 