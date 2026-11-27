"""Google Drive detection on Colab.

Regression test: `setup/colab_setup.sh` used to `mkdir -p /content/drive/MyDrive/...`
before Drive was mounted. The resulting plain folder passed the old "does
/content/drive/MyDrive exist?" check, so every stage saved to Colab's temporary disk
and the work vanished on reconnect. Only a real mount may count as Drive.
"""
import sys

import pytest

from mindeye_lora import env


@pytest.fixture
def colab(monkeypatch):
    monkeypatch.setattr(env, "in_colab", lambda: True)
    monkeypatch.delitem(sys.modules, "ipykernel", raising=False)   # as under `!python`


def test_a_plain_folder_named_mydrive_is_not_drive(tmp_path):
    (tmp_path / "MyDrive").mkdir()
    assert not env.drive_mounted(str(tmp_path))


def test_off_colab_nothing_is_mounted_or_required(monkeypatch):
    monkeypatch.setattr(env, "in_colab", lambda: False)
    assert env.mount_drive() is False


def test_unmounted_drive_stops_with_instructions(tmp_path, colab):
    with pytest.raises(SystemExit, match=r"drive\.mount\('/content/drive'\)"):
        env.mount_drive(str(tmp_path))


def test_the_stub_folder_is_called_out(tmp_path, colab):
    (tmp_path / "MyDrive").mkdir()
    with pytest.raises(SystemExit, match="plain local folder"):
        env.mount_drive(str(tmp_path))


def test_a_real_mount_is_accepted(tmp_path, colab, monkeypatch):
    monkeypatch.setattr(env.os.path, "ismount", lambda p: p == str(tmp_path))
    assert env.mount_drive(str(tmp_path)) is True


def test_default_root_ignores_a_stub_mydrive(colab, monkeypatch, tmp_path):
    monkeypatch.delenv("MINDEYE_LORA_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env, "drive_mounted", lambda *a: False)
    assert env.default_root() == tmp_path / "workspace"
    monkeypatch.setattr(env, "drive_mounted", lambda *a: True)
    assert env.default_root() == env.Path("/content/drive/MyDrive/mindeye2_lora")


def test_no_drive_skips_the_check(tmp_path, colab):
    ws = env.setup_environment(tmp_path / "ws", use_drive=False, min_free_gb=0)
    assert ws.root == tmp_path / "ws"
