"""실제 API나 실제 키 없이 Cloud/로컬 키 읽기와 대화 분리를 검사합니다."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


with patch.dict(os.environ, {"OPENAI_API_KEY": "environment-placeholder"}), patch.object(app.st, "secrets", {"OPENAI_API_KEY": "secrets-placeholder"}):
    assert app.read_api_key() == "environment-placeholder"
with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), patch.object(app.st, "secrets", {"OPENAI_API_KEY": "secrets-placeholder"}):
    assert app.read_api_key() == "secrets-placeholder"
with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), patch.object(app.st, "secrets", {}), patch.object(app, "dotenv_values", return_value={"OPENAI_API_KEY": "dotenv-placeholder"}):
    assert app.read_api_key() == "dotenv-placeholder"
with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), patch.object(app.st, "secrets", {}), patch.object(app, "dotenv_values", return_value={}):
    assert app.read_api_key() == ""
with patch.object(app, "ROOT", Path("nonexistent-cloud-config-fixture")):
    assert not app.persist_chat_history()
with patch.object(Path, "exists", return_value=True):
    assert app.persist_chat_history()
print("PASS: environment / Cloud Secrets / local .env / missing key / Cloud session history")
