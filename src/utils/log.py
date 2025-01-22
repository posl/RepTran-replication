import os, sys
import logging

BASE_LOGGER_NAME = "base_logger"


def set_exp_logging(exp_dir, exp_name, log_file_name=None):
    """訓練や評価など時間かかるので実験番号とか残しておきたい系のログの設定.

    Args:
        exp_dir (str): 実験のディレクトリ.
        exp_name (str): 実験名（識別子）.
        log_file_name (str): log fileの名前 (拡張子つき). 実験の識別子と別で特に指定したい場合に使う.

    Returns:
        logger (logging.logger): logger object.
    """
    if log_file_name is None:
        # exp_nameに.logつけたのをlog file名にする
        log_file_name = f"{exp_name}.log"
    # log_file_nameを引数に与えた場合でも.logがついてるか一応チェック
    else:
        log_file_name = log_file_name + ".log" if not log_file_name.endswith(".log") else log_file_name
    # base_loggerという名前のloggerを作成
    logger = logging.getLogger(BASE_LOGGER_NAME)
    # 表示レベルをINFOに変更 (=DEBUGは表示されない)
    logger.setLevel(logging.INFO)
    # logs dirがない場合は作成
    if not os.path.exists(os.path.join(exp_dir, "logs")):
        os.makedirs(os.path.join(exp_dir, "logs"))
    # handlerとformatterの設定
    fh = logging.FileHandler(filename=os.path.join(exp_dir, "logs", log_file_name))
    formatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # 最初のログ出力
    logger.info(f"Start {sys.argv[0]}")
    logger.info(f"exp_name={exp_name}, exp_dir={exp_dir}, log_file_name={log_file_name}")
    return logger
