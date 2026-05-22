from pathlib import Path

from vizQA.app.logger import configure_logging, get_logger, reset_logger


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


def test_logger_does_not_create_file_until_first_write(tmp_path, monkeypatch):
    monkeypatch.setattr("vizQA.app.logger._LOG_DIR", str(tmp_path))
    reset_logger()

    logger = get_logger()

    assert not Path(logger.log_path).exists()

    logger.log_session("session-1", "start", "url='http://example.com'")
    reset_logger()

    assert Path(logger.log_path).exists()


def test_log_perception_compact_top_matches_and_selected(tmp_path, monkeypatch):
    monkeypatch.setattr("vizQA.app.logger._LOG_DIR", str(tmp_path))
    reset_logger()

    logger = get_logger()
    response = {
        "top_matches": [
            {
                "text": "Sign in",
                "label": "primary-cta",
                "salience": 0.91,
                "similarity": 0.88,
                "bounds": [10, 20, 50, 60],
                "spatial": {"position": "top-right"},
            },
            {
                "placeholder": "Email",
                "salience": 0.44,
                "similarity": 0.55,
                "location": [0.10, 0.20, 0.30, 0.40],
            },
        ]
    }

    logger.log_perception("step-1", "sign in button", response, selected=response["top_matches"][0])
    reset_logger()

    lines = Path(logger.log_path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert (
        "query='sign in button' candidate=#1[src=top_matches text='Sign in' pos=top-right sal=0.91 sim=0.88 geom=b(10,20,50,60)]"
        in lines[0]
    )
    assert (
        "query='sign in button' candidate=#2[src=top_matches text='Email' pos=- sal=0.44 sim=0.55 geom=loc(0.10,0.20,0.30,0.40)]"
        in lines[1]
    )
    assert "selected=#1[src=top_matches text='Sign in' pos=top-right sal=0.91 sim=0.88 geom=b(10,20,50,60)]" in lines[2]


def test_log_perception_falls_back_to_elements_and_handles_missing_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr("vizQA.app.logger._LOG_DIR", str(tmp_path))
    reset_logger()

    logger = get_logger()
    response = {
        "elements": [
            {"label": "username-field"},
            {"name": "password_input", "spatial": {"position": "middle-left"}},
        ]
    }

    logger.log_perception("step-2", "credentials", response, selected=None)
    reset_logger()

    lines = Path(logger.log_path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert "query='credentials' candidate=#1[src=elements text='username-field' pos=- sal=- sim=- geom=-]" in lines[0]
    assert (
        "query='credentials' candidate=#2[src=elements text='password_input' pos=middle-left sal=- sim=- geom=-]"
        in lines[1]
    )
    assert "selected=none" in lines[2]


def test_configure_logging_hides_debug_entries_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr("vizQA.app.logger._LOG_DIR", str(tmp_path))
    reset_logger()
    configure_logging(debug_enabled=False)

    logger = get_logger()
    logger.log_perception("step-3", "search box", {"elements": [{"text": "Search"}]}, selected=None)
    logger.log_session("session-1", "start", "url='http://example.com'")
    reset_logger()

    lines = Path(logger.log_path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "SESSION" in lines[0]
