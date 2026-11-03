import logging, json, warnings
from sklearn.exceptions import DataConversionWarning

_initialized = False
_mocking = False

class Logger:

    ##################################################################
    ##################################################################

    def  __init__(self, name: str):

        # STOP VERBOSE NAME PREFIXES
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        for prefix in ['source.', 'components.']:
            if name.startswith(prefix):
                name = name[len(prefix):]

        self._logger: logging.Logger = logging.getLogger(
            str(name)
        )

        global _initialized

        if not _initialized:
            self._initialize()

    ##################################################################
    ##################################################################

    def info(self, msg: str):
        msg = msg if not self.mocking() else f'[MOCK] {msg}'
        self._logger.info(msg, stacklevel=2)

    ##################################################################
    ##################################################################

    def debug(self, msg: str):
        msg = msg if not self.mocking() else f'[MOCK] {msg}'
        self._logger.debug(msg, stacklevel=2)

    ##################################################################
    ##################################################################

    def critical(self, msg: str):
        msg = msg if not self.mocking() else f'[MOCK] {msg}'
        self._logger.critical(msg, stacklevel=2)

    ##################################################################
    ##################################################################

    def warning(self, msg: str):
        msg = msg if not self.mocking() else f'[MOCK] {msg}'
        self._logger.warning(msg, stacklevel=2)

    ##################################################################
    ##################################################################

    def error(self, result):
        try:
            err = result.error

            if ('problem' in result.error['details']) and result.error['details']['problem'] == 'A mandatory unittest has failed':
                err = result.error['details']['error']

            self._logger.error(
                f'Encountered fatal error: { json.dumps(err, indent=4) }',
                stacklevel=2
            )
        except:
            self._logger.error(
                f'Logger could not parse fatal error',
                stacklevel=2
            )
            print(result)

    ##################################################################
    ##################################################################

    def _initialize(self):
        formatter = _get_formatter()

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # IGNORE TA INDEXING WARNING
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module=r"ta\..*",
        )

        warnings.filterwarnings(
            "ignore",
            category=DataConversionWarning
        )

        # TURN OFF AUXILLARY LOGGERS FOR PINGING SERVICE
        blacklisted_libs = [
            'urllib3',
            'urllib3.connectionpool',
            # 'requests',
            'git',
            'mlflow',
            # 'matplotlib.font_manager',
            # 'matplotlib.pyplot',
            'ta.trend'
        ]

        for lib_name in blacklisted_libs:
            logging.getLogger(lib_name).setLevel(logging.ERROR)

        global _initialized
        _initialized = True

    ##################################################################
    ##################################################################

    def save_logs(self, exp_dir: int):
        formatter = _get_formatter(False)

        # # LOGGING FORMATS
        # log_format = "[%(asctime)s.%(msecs)03d] %(levelname).4s [%(name)s] %(message)s"
        # formatter = logging.Formatter(log_format, datefmt="%d-%m %H:%M:%S")

        # Get root logger
        root_logger = logging.getLogger()

        logfile_path = f'{ exp_dir }/interpreter.log'
        file_handler = logging.FileHandler(logfile_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # COMMUNICATE FILE CREATION
        root_logger.debug(f'Now writing to logfile')

    ##################################################################
    ##################################################################

    def mocking(self):
        global _mocking
        return _mocking

    ##################################################################
    ##################################################################

    def toggle_mocking(self, status: bool):
        global _mocking
        _mocking = status

    ##################################################################
    ##################################################################

def _get_formatter(with_color: bool = True):

    # LOGGING FORMATS
    log_format = "[%(asctime)s.%(msecs)03d] %(levelname).4s [%(name)s] %(message)s"
    logging.Formatter(log_format, datefmt="%d-%m %H:%M:%S")

    RESET = "\x1b[0m" if with_color else ''
    DIM   = "\x1b[2m" if with_color else ''

    COLORS = {
        "DEBUG": "\x1b[36m" if with_color else '',   # Cyan
        "INFO": "\x1b[32m" if with_color else '',    # Green
        # "INFO": "\x1b[48;2;0;128;0m\x1b[38;2;255;255;255m" if with_color else '',    # Green
        "WARNING": "\x1b[33m" if with_color else '', # Yellow
        "ERROR": "\x1b[41m\x1b[97m" if with_color else '',     # Red background, white text
        "CRITICAL": "\x1b[45;1m\x1b[97m" if with_color else '', # Bright magenta background, white text
        "MOCK": "\x1b[38;5;218m" if with_color else ''
    }

    short_levels = {
        "DEBUG": "DEBU",
        "INFO": "INFO",
        "WARNING": "WARN",
        "ERROR": "ERRO",
        "CRITICAL": "CRIT",
    }

    class ColorFormatter(logging.Formatter):
        def format(self, record):
            short_name = short_levels.get(record.levelname, record.levelname[:4])

            # Apply color to level
            if record.levelname in COLORS:
                record.levelshort = f"{COLORS[record.levelname]}{short_name}{RESET}"
            else:
                record.levelshort = short_name

            # Build timestamp separately, wrapped in DIM
            timestamp = self.formatTime(record, "%d-%m %H:%M:%S")
            timestamp = f"{DIM}[{timestamp}.{record.msecs:03.0f}]{RESET}"

            mocking_status = False
            msg = record.getMessage()

            if '[MOCK]' in msg:
               msg = msg.replace('[MOCK] ', '')
               mocking_status = True

            # NOT MOCKING, OUTPUT NORMALLY
            if not mocking_status:
                return f"{timestamp} {record.levelshort} [{record.name}] {msg}"

            # OTHERWISE, REPLACE DEBUG LEVEL WITH MOCK FLAIRE
            # mock_text = f"{DIM}-{RESET}{COLORS['MOCK']} MOCK{RESET}"
            # return f"{timestamp} {record.levelshort} { mock_text } [{record.name}] {msg}"
            mock_text = f"{COLORS['MOCK']}MOCK{RESET}"
            return f"{timestamp} { mock_text } [{record.name}] {msg}"
        
    return ColorFormatter()