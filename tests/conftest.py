import json
import random
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _seed():
    random.seed(0)
    np.random.seed(0)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
