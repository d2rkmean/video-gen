import logging
import time
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.progress import track
from rich.traceback import install

console = Console(stderr=True)


class InterceptHandler(logging.Handler):
    """Перенаправляет стандартные логи (urllib3 и др.) в Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        frame = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            if frame.f_back is None:
                break
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


class _LoggerState:
    console_sink_id: int | None = None
    target_console: Console | None = None
    console_level: str = "INFO"


_state = _LoggerState()

logger.level("INFO", icon="I")
logger.level("WARNING", icon="W")
logger.level("ERROR", icon="E")
logger.level("DEBUG", icon="D")
logger.level("SUCCESS", icon="S")
logger.level("CRITICAL", icon="C")


def logcat_formatter(record):
    worker_name = record["extra"].get("worker", record["name"])
    lvl = record["level"].name
    msg = record["message"]

    styles = {
        "DEBUG": ("black on white", "dim white"),
        "INFO": ("bold white on blue", "white"),
        "SUCCESS": ("bold black on green", "bold green"),
        "WARNING": ("bold black on yellow", "bold yellow"),
        "ERROR": ("bold white on red", "bold red"),
        "CRITICAL": ("bold white on magenta", "bold magenta"),
    }
    badge_style, _ = styles.get(lvl, ("white", "white"))

    # Стилизация ключевых тегов стадий конвейера
    tag_styles = {
        "[CACHE HIT]": "[bold black on green] CACHE HIT [/bold black on green]",
        "[CACHE MISS]": "[bold white on dark_orange] CACHE MISS [/bold white on dark_orange]",
        "[YOUTUBE]": "[bold white on red] YOUTUBE [/bold white on red]",
        "[SPONSOR]": "[bold black on yellow] SPONSOR [/bold black on yellow]",
        "[PYGAME]": "[bold white on dark_cyan] PYGAME [/bold white on dark_cyan]",
    }
    for tag, styled_tag in tag_styles.items():
        if tag in msg:
            msg = msg.replace(tag, styled_tag)

    return (
        f"[dim][{{time:DD.MM.YYYY HH:mm:ss}}][/dim]  "
        f"[{badge_style}] {{level.icon}} [/{badge_style}]  "
        f"[bold cyan]{worker_name}[/bold cyan]: {msg}\n"
    )


def setup_logger(
    log_dir: str | Path = "logs",
    out_console: Console | None = None,
    console_level: str = "INFO",
) -> None:
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(exist_ok=True, parents=True)

    _state.target_console = out_console or console
    _state.console_level = console_level
    logger.remove()

    tc = _state.target_console or console
    _state.console_sink_id = logger.add(
        lambda msg: tc.print(msg, end=""),
        format=logcat_formatter,
        level=console_level,
        enqueue=True,
        colorize=False,
    )

    log_file_path = log_dir_path / "farm_{time:YYYY-MM-DD}.log"
    logger.add(
        str(log_file_path),
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        format="[{time:DD.MM.YYYY HH:mm:ss}] [{level.name:<7}] {name}: {message}\n",
        level="DEBUG",
        enqueue=True,
        encoding="utf-8",
    )

    install(show_locals=False)

    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)

    logger.debug("Логгер инициализирован.")


if __name__ == "__main__":
    setup_logger()
    logger.info("INFO TEST")
    logger.warning("WARNING TEST")
    logger.error("ERROR TEST")

    for i in track(range(4), description="PROGRESSBAR TEST", console=console):
        time.sleep(1)
        if i == 2:
            logger.info("IN-PROGRESSBAR LOG TEST!")
