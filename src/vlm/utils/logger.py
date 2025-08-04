import logging
import datetime
from pathlib import Path

LOG_FMT1 = "%(asctime)s:%(name)s:%(levelname)s:%(message)s"


class NoTracebackFormatter(logging.Formatter):

    def __init__(self, fmt: str | None = None, *args, **kwargs) -> None:
        super().__init__(fmt, *args, **kwargs)

    def formatException(self, exc_info):
        return '%s: %s' % (exc_info[0].__name__, exc_info[1])
    
    def formatStack(self, stack_info):
        return ""


class CustomLogger(logging.Logger):
    """
    Custom logger class that support logging to local file and GCS.The output file name would be `{current_timestamp}_{source_name}.log`
    ----------
    Args:
        source_name: str
            Name of the flow file that you want to log, suggest to leave it as __name__ . This helps naming the log file
        level: Optional[str]
            Log level, default to INFO
        log_to_local: Optional[bool]
            Whether to write log to local file, default to False
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Create a singleton. Return the existed client if one's been already initiated
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        logger_name: str = "logger",
        level: str = "INFO",
        log_to_local: bool = False,
        log_path: str = "./logs",
        disable_traceback = False,
        *args,
        **kwargs,
    ):
        super().__init__(logger_name, level, *args, **kwargs)
        self.disable_traceback = disable_traceback
        self.log_path = Path(log_path)

        self._add_stream_handler()

        if log_to_local:
            self._add_file_handler()

    def _add_stream_handler(self):
        """Add stream handler to stream logging info to the console"""
        stream_handler = logging.StreamHandler()
        formatter = NoTracebackFormatter(LOG_FMT1) if self.disable_traceback else logging.Formatter(LOG_FMT1)
        stream_handler.setFormatter(formatter)
        self.addHandler(stream_handler)

    def _add_file_handler(self):
        """Add file handler to write log to local file """
        if not self.log_path.exists():
            self.log_path.mkdir(parents=True)

        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{self.name}_{now}.log"

        file_handler = logging.FileHandler(filename=self.log_path / log_filename)
        formatter = NoTracebackFormatter(LOG_FMT1) if self.disable_traceback else logging.Formatter(LOG_FMT1)
        file_handler.setFormatter(formatter)
        self.addHandler(file_handler)
