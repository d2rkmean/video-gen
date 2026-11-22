#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
cd "$DIR"

NVIDIA_LIBS=""
for d in "$DIR"/.venv/lib/python*/site-packages/nvidia/*/lib; do
    if [ -d "$d" ]; then
        NVIDIA_LIBS="${NVIDIA_LIBS:+$NVIDIA_LIBS:}$d"
    fi
done
if [ -n "$NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

DO_POWEROFF=false
PASSED_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--poweroff" ] || [ "$arg" = "-p" ]; then
        DO_POWEROFF=true
    else
        PASSED_ARGS+=("$arg")
    fi
done

if [ "$DO_POWEROFF" = true ]; then
    echo "Режим автовыключения после завершения включён."
    "$DIR/.venv/bin/video-gen" "${PASSED_ARGS[@]}" && {
        echo "Генерация завершена. Выключение системы через 20 секунд (Ctrl+C для отмены)..."
        sleep 20
        systemctl poweroff || poweroff
    }
else
    exec "$DIR/.venv/bin/video-gen" "${PASSED_ARGS[@]}"
fi
