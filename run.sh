#!/bin/bash
# 语义知识树  v0.3  —  持久化 + phi4-mini  +  日志记录
cd "$(dirname "$0")"
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama 未运行 — 推理层将使用规则引擎 fallback"
    echo "   运行: ollama serve"
    echo ""
fi

echo "======================================================="
echo "  🌐 全球语义知识树系统  v0.3"
echo "======================================================="
echo ""
echo "  1 — 批量演示（6 个预设查询）"
echo "  2 — 交互模式（自由输入）"
echo "  3 — 机制验证测试"
echo "  4 — 查看日志"
echo ""

read -p "选择 [1/2/3/4]: " mode

case "$mode" in
  3)
    exec python test_core.py
    ;;
  4)
    f=$(ls -1t logs/session-*.jsonl 2>/dev/null | head -1)
    if [ -z "$f" ]; then
        echo "暂无日志文件"; exit 0
    fi
    echo "📋 最近的日志: $f"
    echo ""
    exec python3 -c "
import json
with open('$f') as fh:
    lines = [json.loads(l) for l in fh if l.strip()]
for r in lines:
    print(f'[#{r[\"seq\"]}] {r[\"query\"]}  ({r[\"intent\"]})')
    print(f'    渗透 {len(r[\"penetration\"])} 条, 捷径 {len(r[\"shortcut\"])} 条')
    for p in r['penetration'][:3]:
        print(f'      渗透  w={p[\"total_weight\"]}  {p[\"leaf\"]}')
    for s in r['shortcut'][:3]:
        print(f'      捷径  s={s[\"similarity\"]}  {s[\"leaf\"]}')
    print()
"
    ;;
  *)
    # 批量(1) 和交互(2) 都由 demo.py 处理
    exec python demo.py
    ;;
esac
