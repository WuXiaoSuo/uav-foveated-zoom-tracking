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
COMMIT_MESSAGE="${1:-保存 UFZ-Track 项目进展}"
TRACKED_PATHS=(
    "README.md"
    ".gitignore"
    "configs"
    "scripts"
    "src"
    "docs"
    "tools"
)

echo "项目目录：$PROJECT_DIR"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "错误：项目目录不存在。请进入 Git 仓库后运行，或设置 UFZTRACK_PROJECT_DIR。"
    exit 1
fi

cd "$PROJECT_DIR" || exit 1

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "错误：当前项目目录不是 Git 仓库，无法保存。"
    exit 1
fi

echo "仅暂存代码、配置、脚本、文档和工具目录。"
git add -- "${TRACKED_PATHS[@]}"

if git diff --cached --quiet; then
    echo "没有需要提交的白名单改动。"
    exit 0
fi

echo "即将提交：$COMMIT_MESSAGE"
git commit -m "$COMMIT_MESSAGE"
echo "保存完成。"
