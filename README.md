<div align="center">

# video-gen

**Асинхронный CLI-конвейер для пакетной генерации вертикальных роликов (Shorts / Reels / TikTok / VK Клипы)**

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Ruff](https://img.shields.io/badge/Linter-Ruff-D7FF64?style=flat-square&logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![Mypy](https://img.shields.io/badge/Typecheck-Mypy-2A6DB2?style=flat-square)](https://mypy-lang.org/)
[![FFmpeg NVENC](https://img.shields.io/badge/Render-FFmpeg%20%2F%20NVENC-007808?style=flat-square&logo=ffmpeg&logoColor=white)](https://ffmpeg.org/)

*Автоматический пайплайн от исходного видео или ссылки YouTube до готового вертикального клипа с распознаванием речи, поиском лучших моментов через LLM, трекингом лица, караоке-субтитрами и GPU-ускорением.*

**Графический интерфейс (Desktop & Web):** [`video-gen-gui`](https://github.com/d2rkmean/video-gen-gui)

</div>

---

## Как устроен конвейер

```text
  Источники (файлы / YouTube via yt-dlp + SponsorBlock)
         │
         ▼
  Транскрипция речи (faster-whisper CUDA/CPU или Groq / OpenAI API + дисковый кэш)
         │
         ▼
  Сегментация (LLM highlights через OpenAI-совм. API  или  speech_split по паузам)
         │
         ├──► Анимированные ASS-субтитры (караоке \kf, pop / bounce)
         ├──► Трекинг лица спикера (OpenCV Haar Cascade) + нижний фон (видео / PyGame)
         └──► Музыка с auto-ducking + баннеры (overlay / center / interval)
         │
         ▼
  Параллельный рендер FFmpeg (h264_nvenc / h265_nvenc / libx264 / libx265)
         │
         ▼
  Готовые .mp4 + .json метаданные (+ опциональная каталогизация в PostgreSQL)
```

---

## Возможности

- **Источники контента**:
  - локальные файлы и директории (`.mp4`, `.mkv`, `.mov`, `.webm`, `.avi`);
  - одиночные видео и плейлисты YouTube через `yt-dlp`.
- **Транскрипция речи**:
  - локальный `faster-whisper` (`CUDA` / `CPU`) или облачный API (Groq, OpenAI);
  - автоматический выбор целевой аудиодорожки в многодорожечных контейнерах;
  - дисковое кэширование результатов транскрипции.
- **Умная нарезка клипов**:
  - `highlights` — отбор лучших фрагментов через любой OpenAI-совместимый API (**OpenAI**, **Groq**, **Google Gemini OpenAI endpoint**, **OpenRouter**, **Ollama**) с генерацией заголовков, хуков и описаний;
  - `speech_split` — последовательная нарезка по границам предложений и естественным паузам без обращений к LLM;
  - интеграция со **SponsorBlock API** (для YouTube) и эвристический фильтр рекламных вставок.
- **Нижний фон**:
  - случайные или последовательные фоновые ролики из директории;
  - **headless-генератор процедурного фона на PyGame** с пресетами `runner`, `particles`, `matrix` (работает без X-сервера).
- **Рендеринг FFmpeg**:
  - аппаратное ускорение `h264_nvenc` / `h265_nvenc` и CPU-кодирование `libx264` / `libx265`;
  - параллельный рендеринг (`params.concurrent_renders`, от 1 до 4 воркеров);
  - трекинг лица спикера (OpenCV Haar Cascade), настройка пропорций верх/низ, цветокоррекция, резкость, микро-рандомизация и очистка метаданных;
  - баннеры с режимами `center`, `end`, `overlay`, `interval_minutes`, `interval_percent`, стоп-кадром, размытием, хромакеем и автообрезкой;
  - фоновая музыка с автоприглушением голоса (ducking) и плавным затуханием.
- **Субтитры ASS**:
  - пословная караоке-подсветка (`\kf`), анимации `pop` / `bounce` / `none`, настройка шрифта, цветов (`&HAABBGGRR`), обводки и позиции.
- **Каталогизация и GUI**:
  - опциональное сохранение иерархических кодов фильмов, серий и нарезок (`XXXXX`, `XXXXX-серия-клип`) в **PostgreSQL** (`asyncpg`);
  - структурированный поток прогресса в `stdout` (`--json-progress`) для интеграции с [`video-gen-gui`](https://github.com/d2rkmean/video-gen-gui).

---

## Требования

- **ОС:** Linux (рекомендуется для CUDA / NVENC), macOS или WSL2
- **Python:** 3.12+
- **Менеджер пакетов:** [uv](https://docs.astral.sh/uv/)
- **Системные утилиты:** `ffmpeg` и `ffprobe` в `PATH`
- **GPU (опционально):** видеокарта NVIDIA с драйвером CUDA и поддержкой `h264_nvenc` в FFmpeg

---

## Установка и быстрый старт

```bash
git clone https://github.com/d2rkmean/video-gen.git
cd video-gen
uv sync
```

1. Скопируйте шаблон конфигурации:
   ```bash
   cp config.json.example config.json
   ```
2. Положите исходные видео в `input/`, фоновые ролики в `bottom/` (или укажите `"type": "pygame"`), шрифты `.ttf`/`.otf` в `fonts/`.
3. Задайте API-ключ для LLM-анализа (если используется `mode = "highlights"`):
   ```bash
   export OPENAI_API_KEY="sk-..."
   # либо GEMINI_API_KEY / GROQ_API_KEY, или укажите analysis_api_key в config.json
   ```
4. Запустите конвейер:
   ```bash
   uv run video-gen config.json
   ```

Если CUDA-библиотеки `cuBLAS`/`cuDNN` установлены внутри `.venv`, можно запускать через скрипт-обёртку (автоматически выставляет `LD_LIBRARY_PATH`):

```bash
./run_farm.sh config.json
```

---

## Использование CLI

```text
video-gen [config.json] [--json-progress]
```

| Параметр / переменная | Описание |
| :--- | :--- |
| `config.json` | Путь к конфигурации (поддерживается `.json` и legacy `.toml`). По умолчанию `config.json`. |
| `--json-progress` | Вывод событий жизненного цикла конвейера в `stdout` в формате JSON Lines. |
| `VIDEOGEN_JSON_PROGRESS=1` | Эквивалент флага `--json-progress` через переменную окружения. |
| `-h`, `--help` | Вывести справку. |

**Коды выхода:**
* `0` — конвейер успешно завершён;
* `1` — нет доступных источников или критическая ошибка выполнения;
* `2` — неверные аргументы CLI, отсутствующий или невалидный файл конфигурации;
* `130` — остановка пользователем (`SIGINT`).

---

## Конфигурация (`config.json`)

Схема конфигурации валидируется строго через Pydantic (`extra="forbid"`): опечатка в имени поля останавливает запуск до начала обработки.

| Секция | Назначение |
| :--- | :--- |
| `videos.top_videos` | Основные источники (`type`: `"video"` или `"youtube"`). Поддерживают переопределение `output_dir`, `mode`, `parts_count`, `part_duration`, `codename`, `base_code`, `season`, `episode` и вложенных секций. |
| `videos.bottom_videos` | Фоновые источники (`type`: `"video"` или `"pygame"` с пресетами `"runner"`, `"particles"`, `"matrix"`). |
| `params` | Транскрипция (`transcribe_backend`, `whisper_model`, `whisper_device`, `language`), анализ (`analysis_backend`, `analysis_api_base`, `analysis_model`), режим нарезки (`highlights` / `speech_split`), SponsorBlock, `concurrent_renders` и флаги кэша. |
| `assets` | Пути к `banner`, `banners_dir`, `music_dir`, `fonts_dir`. |
| `render` | Разрешение (`width`, `height`, `fps`), пропорции (`top_parts`, `bottom_parts`), кодек (`h264_nvenc`, `libx264` и др.), трекинг лиц, цветокоррекция, антидетект, параметры баннера и звука. |
| `subtitles` | `enabled`, `font`, `font_size`, цвета ASS `&HAABBGGRR`, анимация (`pop`, `bounce`, `none`), лимиты строки (`max_words`, `max_chars`). |
| `database` | Опциональное подключение к PostgreSQL (`enabled`, `host`, `port`, `user`, `password`, `database`). |

**Переменные окружения** (подхватываются автоматически при пустых полях в конфиге):
* `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`
* `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`

---

## Структура проекта

```text
video-gen/
├── src/video_gen/
│   ├── __main__.py          # Точка входа CLI, оркестрация и --json-progress
│   ├── config.py            # Строгие Pydantic-модели конфигурации
│   ├── models.py            # Модели сегментов, слов, хайлайтов и метаданных
│   ├── callbacks.py         # Коллбэки прогресса и событий конвейера
│   ├── gameplay/            # Headless-генератор фонов на PyGame
│   ├── tasks/               # Задачи: youtube, transcribe, best_parts, subtitle, ffmpeg, film_worker
│   └── utils/               # Face tracker, ad filter, audio utils, speech splitter, cache
├── scripts/                 # Вспомогательные скрипты (генерация баннеров, выгрузка)
├── tests/                   # Юнит- и интеграционные тесты (pytest)
├── config.json.example      # Пример конфигурации
└── pyproject.toml           # Зависимости и настройки ruff / mypy / pytest
```

---

## Тестирование и качество кода

```bash
uv run ruff check src tests
uv run mypy src tests
uv run pytest -v
```

Запуск быстрых тестов без тяжёлого рендеринга FFmpeg:

```bash
uv run pytest -m "not ffmpeg" -v
```

---

## Лицензия

Проект распространяется под лицензией [MIT](LICENSE).
