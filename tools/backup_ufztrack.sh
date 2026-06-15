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
BACKUP_ROOT="${UFZTRACK_BACKUP_DIR:-/tmp/ufztrack_backups}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_FILE="$BACKUP_ROOT/ufztrack_code_backup_$TIMESTAMP.tar.gz"
BACKUP_PATHS=(
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
mkdir -p "$BACKUP_ROOT"

echo "备份目录：$BACKUP_ROOT"
echo "只备份代码、配置、脚本、文档和工具目录，不备份 data、outputs、weights。"

tar -czf "$BACKUP_FILE" "${BACKUP_PATHS[@]}"

echo "备份完成：$BACKUP_FILE"
