from __future__ import annotations

from pathlib import Path

from medarc_rl.utils import create_job_cache_root


def test_create_job_cache_root_uses_unique_directory_under_slurm_tmpdir(tmp_path: Path) -> None:
    first = create_job_cache_root("47069", str(tmp_path))
    second = create_job_cache_root("47069", str(tmp_path))

    assert first.parent == tmp_path
    assert second.parent == tmp_path
    assert first != second
    assert first.name.startswith("medarc-rl-47069-")
    assert second.name.startswith("medarc-rl-47069-")
    assert first.exists()
    assert second.exists()


def test_create_job_cache_root_falls_back_to_system_tmp_without_shared_medarc_dir(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("medarc_rl.utils.tempfile.gettempdir", lambda: str(tmp_path))

    cache_root = create_job_cache_root("job/with spaces", None)

    assert cache_root.parent == tmp_path
    assert cache_root.name.startswith("medarc-rl-job_with_spaces-")
    assert cache_root.exists()
    assert not (tmp_path / "medarc").exists()
