#!/bin/bash
cd /data/semantic-knowledge-tree
source .venv/bin/activate
python3 -c "
from core.reasoning import MODEL
print('MODEL=', MODEL)
"
ollama list 2>/dev/null | grep phi4
echo "---"
ollama ps 2>/dev/null | head -3
