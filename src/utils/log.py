import os, sys
import logging

BASE_LOGGER_NAME = "base_logger"


def set_exp_logging(exp_dir, exp_name, log_file_name=None):
    """Logging setup for experiments where we want to keep experiment numbers etc. because training and evaluation take time.

    Args:
        exp_dir (str): Experiment directory.
        exp_name (str): Experiment name (identifier).
        log_file_name (str): Log file name (with extension). Used when you want to specify it separately from the experiment identifier.

    Returns:
        logger (logging.logger): logger object.
    """
    if log_file_name is None:
        # Use exp_name with .log appended as log file name
        log_file_name = f"{exp_name}.log"
    # Check if .log is attached even when log_file_name is given as argument
    else:
        log_file_name = log_file_name + ".log" if not log_file_name.endswith(".log") else log_file_name
    # Create logger with name base_logger
    logger = logging.getLogger(BASE_LOGGER_NAME)
    # Change display level to INFO (=DEBUG is not displayed)
    logger.setLevel(logging.INFO)
    # Create logs dir if it doesn't exist
    if not os.path.exists(os.path.join(exp_dir, "logs")):
        os.makedirs(os.path.join(exp_dir, "logs"))
    # Set up handler and formatter
    fh = logging.FileHandler(filename=os.path.join(exp_dir, "logs", log_file_name))
    formatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # First log output
    logger.info(f"Start {sys.argv[0]}")
    logger.info(f"exp_name={exp_name}, exp_dir={exp_dir}, log_file_name={log_file_name}")
    return logger
