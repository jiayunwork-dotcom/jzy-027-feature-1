import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture()
def client(tmp_path):
    from app.app import create_app

    flask_app = create_app(data_dir=str(tmp_path / "data"))
    flask_app.config.update(TESTING=True)
    return flask_app.test_client()
