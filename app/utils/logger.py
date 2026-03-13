import logging
import sys
import os
from logging.handlers import TimedRotatingFileHandler


def setup_logger(name: str):
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

        # --- 輸出到終端機 (Console) ---
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 這樣就不會觸發 Windows 的檔案鎖定衝突
        if os.getenv("TESTING") == "true":
            return logger

        # --- 輸出到檔案 (File Handler) ---
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(log_dir, "app.log")
        try:
            file_handler = TimedRotatingFileHandler(log_file,
                                                    when="midnight",
                                                    interval=1,
                                                    backupCount=0,
                                                    encoding="utf-8")
            file_handler.suffix = "%Y-%m-%d"
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # 即使檔案被鎖定，也別讓 Logger 崩潰，至少還有 Console Log
            print(f"Warning: Could not initialize file handler: {e}")

    return logger
