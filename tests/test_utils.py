from __future__ import annotations

import random

from hirm.utils.logging import ExperimentLogger
from hirm.utils.seed import set_seed


def test_set_seed_repeatability() -> None:
    set_seed(5)
    values_one = [random.random() for _ in range(3)]
    set_seed(5)
    values_two = [random.random() for _ in range(3)]
    assert values_one == values_two


def test_logger_writes_file(tmp_path) -> None:
    logger = ExperimentLogger("logger_test", output_dir=tmp_path)
    logger.info("hello world")
    logger.close()
    log_file = tmp_path / "logger_test" / "log.txt"
    assert log_file.exists()
    contents = log_file.read_text(encoding="utf-8")
    assert "hello world" in contents
