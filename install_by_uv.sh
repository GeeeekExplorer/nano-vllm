#!/bin/bash
# 使用 uv 安装并激活 Python 虚拟环境

set -e  # 遇到错误立即退出

echo "================================"
echo "🐍 Python虚拟环境管理"
echo "================================"

# 检查 uv 是否已安装
if ! command -v uv &> /dev/null; then
    echo "❌ uv 未安装"
    echo "📥 正在安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # 重新加载环境变量
    export PATH="$HOME/.cargo/bin:$PATH"

    # 再次检查
    if ! command -v uv &> /dev/null; then
        echo "❌ uv 安装失败，请手动安装："
        echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi

    echo "✅ uv 安装成功"
fi

# 检查虚拟环境是否存在
if [ ! -d ".venv" ]; then
    echo ""
    echo "📦 虚拟环境不存在，正在创建..."
    uv venv
    echo "✅ 虚拟环境创建成功"
else
    echo "✅ 虚拟环境已存在"
fi

# 激活虚拟环境并安装依赖
echo ""
echo "📥 检查并安装依赖..."

# 先激活虚拟环境
source .venv/bin/activate

# 安装依赖
if [ -f "requirements.txt" ]; then
    uv pip install -r requirements.txt
    echo "✅ 依赖(requirements.txt)安装完成"
else
    echo "⚠️  未找到 requirements.txt"
fi

# 安装依赖
if [ -f "pyproject.toml" ]; then
    uv pip install -e .
    echo "✅ 依赖(pyproject.toml)安装完成"
else
    echo "⚠️  未找到 pyproject.toml"
fi

echo ""
echo "================================"
echo "✅ 虚拟环境已激活！"
echo "================================"
echo ""
echo "当前环境信息："
echo "  Python: $(python --version)"
echo "  位置: $VIRTUAL_ENV"
echo ""
echo "已安装的包："
uv pip list
echo ""
echo "================================"
echo "💡 下一步："
echo "================================"
echo ""
echo "1️⃣  激活虚拟环境："
echo "    source .venv/bin/activate"

