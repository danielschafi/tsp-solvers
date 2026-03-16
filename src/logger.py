import logging
import sys
from datetime import datetime
from pathlib import Path

_LOG_FORMAT = (
    "%(asctime)s [%(levelname)-8s] %(filename)s:%(lineno)d (%(funcName)s) — %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_dir: Path = Path("logs"), run_ts: str | None = None) -> None:
    """
    Configure console and file logging for the src package.
    Call once per program invocation, early in main().

    Args:
        log_dir: Directory to write log files to.
        run_ts:  Timestamp string used as the log filename (e.g. "20260303_170601").
                 Defaults to the current time.
    """
    if run_ts is None:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_dir / f"{run_ts}.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Attach to the "src" package logger so all src.* loggers are captured
    # without picking up noise from third-party libraries.
    pkg_logger = logging.getLogger("src")
    pkg_logger.setLevel(logging.INFO)
    pkg_logger.addHandler(console_handler)
    pkg_logger.addHandler(file_handler)
    pkg_logger.propagate = False
