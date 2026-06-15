#!/usr/bin/env bash

set -u

DEFAULT_AUTODL_DIR="/root/autodl-tmp/UFZTrack/code/uav-foveated-zoom-tracking"

resolve_project_dir() {
    if [ -n "${UFZTRACK_PROJECT_DIR:-}" ]; then
        printf "%s\n" "$UFZTRACK_PROJECT_DIR"
        return
    fi

    if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git rev-parse --show-toplevel
        return
    fi

    printf "%s\n" "$DEFAULT_AUTODL_DIR"
}

PROJECT_DIR="$(resolve_project_dir)"

echo "项目目录：$PROJECT_DIR"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "错误：项目目录不存在。请进入 Git 仓库后运行，或设置 UFZTRACK_PROJECT_DIR。"
    exit 1
fi

cd "$PROJECT_DIR" || exit 1

echo "当前工作目录：$(pwd)"

if command -v python3 >/dev/null 2>&1; then
    echo "Python 版本：$(python3 --version 2>&1)"
else
    echo "警告：未检测到 python3，请确认运行环境是否已配置。"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "CUDA 设备："
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "警告：nvidia-smi 可用，但无法读取 GPU 信息。"
else
    echo "警告：未检测到 nvidia-smi 或本地 CUDA 不可用；可以继续进行 CPU/本地调试。"
fi

echo "UFZ-Track 环境检查完成。"
