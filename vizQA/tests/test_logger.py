from pathlib import Path

from vizQA.app.logger import get_logger, reset_logger


def test_get_logger_uses_shared_timestamp_with_viewport_suffixes(tmp_path, monkeypatch):
    monkeypatch.setattr("vizQA.app.logger._LOG_DIR", str(tmp_path))
    reset_logger()

    mobile_logger = get_logger("mobile")
    desktop_logger = get_logger("desktop")

    mobile_name = Path(mobile_logger.log_path).name
    desktop_name = Path(desktop_logger.log_path).name

    assert mobile_name.startswith("run_")
    assert desktop_name.startswith("run_")
    assert mobile_name.endswith("_mobile.log")
    assert desktop_name.endswith("_desktop.log")
    assert mobile_name.replace("_mobile.log", "") == desktop_name.replace("_desktop.log", "")

    reset_logger()
