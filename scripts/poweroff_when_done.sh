#!/usr/bin/env bash
set -u

CANCEL_FILE="/tmp/cancel_video_gen_poweroff"
rm -f "$CANCEL_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ожидание завершения процесса video-gen (PID 104140, 104157)..."

# Ждем пока завершатся оба процесса
while kill -0 104140 2>/dev/null || kill -0 104157 2>/dev/null; do
    if [ -f "$CANCEL_FILE" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Обнаружен файл отмены $CANCEL_FILE. Автовыключение отменено."
        exit 0
    fi
    sleep 5
done

# Дополнительно ждем, если еще работает какой-либо дочерний процесс video_gen
while pgrep -u "$UID" -f "src/video_gen/__main__.py" >/dev/null 2>&1; do
    if [ -f "$CANCEL_FILE" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Обнаружен файл отмены $CANCEL_FILE. Автовыключение отменено."
        exit 0
    fi
    sleep 5
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Процесс video-gen завершён!"

if [ -f "$CANCEL_FILE" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Автовыключение отменено."
    exit 0
fi

# Буфер 20 секунд для синхронизации данных и завершения сопутствующих задач
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ожидание 20 секунд перед выключением ПК..."
for i in {20..1}; do
    if [ -f "$CANCEL_FILE" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Автовыключение отменено во время обратного отсчета."
        exit 0
    fi
    sleep 1
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Синхронизация дисков..."
sync

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Отправка команды на выключение ПК..."
systemctl poweroff || loginctl poweroff
