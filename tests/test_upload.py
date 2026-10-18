import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from upload_model import embed_chat_template  # noqa: E402


def test_chat_template_is_copied_into_tokenizer_config_once(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 131072}))
    (tmp_path / "chat_template.jinja").write_text("{% for m in messages %}{{ m.content }}{% endfor %}")
    embed_chat_template(tmp_path)
    embed_chat_template(tmp_path)  # running it again changes nothing
    cfg = json.loads((tmp_path / "tokenizer_config.json").read_text())
    assert cfg == {"model_max_length": 131072, "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}"}


def test_existing_template_is_left_alone(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "keep me"}))
    (tmp_path / "chat_template.jinja").write_text("other")
    embed_chat_template(tmp_path)
    assert json.loads((tmp_path / "tokenizer_config.json").read_text())["chat_template"] == "keep me"
