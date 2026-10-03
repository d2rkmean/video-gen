import time
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.progress import track
from rich.traceback import install

console = Console(stderr=True)

logger.level("INFO", icon="I")
logger.level("WARNING", icon="W")
logger.level("ERROR", icon="E")
logger.level("DEBUG", icon="D")
logger.level("SUCCESS", icon="S")
logger.level("CRITICAL", icon="C")


def logcat_formatter(record):
    worker_name = record["extra"].get("worker", record["name"])
    lvl = record["level"].name

    styles = {
        "DEBUG": ("black on white", "dim white"),
        "INFO": ("bold white on blue", "bold blue"),
        "SUCCESS": ("bold black on green", "bold green"),
        "WARNING": ("bold black on yellow", "bold yellow"),
        "ERROR": ("bold white on red", "bold red"),
        "CRITICAL": ("bold white on magenta", "bold magenta"),
    }
    badge_style, text_style = styles.get(lvl, ("white", "white"))

    return (
        f"[dim][{{time:DD.MM.YYYY HH:mm:ss}}][/dim]  "
        f"[{badge_style}] {{level.icon}} [/{badge_style}]  "
        f"[{text_style}]{worker_name}: {{message}}[/{text_style}]\n"
    )


def setup_logger(log_dir: str | Path = "logs") -> None:
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(exist_ok=True, parents=True)

    logger.remove()

    logger.add(
        lambda msg: console.print(msg, end=""),
        format=logcat_formatter,
        level="INFO",
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
