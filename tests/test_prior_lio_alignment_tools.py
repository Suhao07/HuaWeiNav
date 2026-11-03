import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from prior_map.alignment import PriorMapAlignment

ROOT = Path(__file__).resolve().parents[1]


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_prior_lio_alignment", ROOT / "scripts/build_prior_lio_alignment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "prior_map.json").write_text(json.dumps({"scene_id": "scene", "frame_id": "map_z_up"}))
    (bundle / "manifest.json").write_text(json.dumps({"map_id": "m", "map_revision": "r"}))
    return bundle


def _args(bundle: Path, **overrides):
    values = dict(bundle=str(bundle), base_xy="1,2", base_yaw_deg=None, heading_xy=None,
                  base_in_lidar="-0.2,0.0,-1.5708", lio_map=None, runtime_frame="camera_init", confidence=1.0)
    values.update(overrides)
    return types.SimpleNamespace(**values)


def test_builder_maps_marker_base_to_static_tf_and_heading_to_base_forward(tmp_path):
    builder = _load_builder()
    alignment, check = builder.build_alignment(_args(_bundle(tmp_path), heading_xy="1,3"))

    assert check["base_marker_in_runtime"] == pytest.approx(check["expected"], abs=1e-9)
    # base 朝 prior +y；base 前方 1 m 的点在雷达系 = R(phi)(1,0) + t。
    phi = -1.5708
    expected_ahead = (-0.2 + math.cos(phi), math.sin(phi))
    assert alignment.prior_to_runtime((1.0, 3.0))[:2] == pytest.approx(expected_ahead, abs=1e-6)
    assert alignment.runtime_frame_id == "camera_init"
    assert alignment.prior_frame_id == "map_z_up"
    assert alignment.diagnostics["anchor"]["type"] == "start_marker"
    assert alignment.diagnostics["prior_scene_id"] == "scene"


def test_builder_records_lio_map_anchor_and_rejects_sha_mismatch(tmp_path):
    builder = _load_builder()
    bundle = _bundle(tmp_path)
    (tmp_path / "lio_map.pcd").write_bytes(b"pcd")
    meta = tmp_path / "lio_map.json"
    meta.write_text(json.dumps({"pcd": "lio_map.pcd", "sha256": hashlib.sha256(b"pcd").hexdigest()}))

    alignment, _ = builder.build_alignment(_args(bundle, base_yaw_deg=30.0, lio_map=str(meta)))
    assert alignment.diagnostics["anchor"]["type"] == "lio_map"
    assert alignment.diagnostics["anchor"]["lio_map_sha256"] == hashlib.sha256(b"pcd").hexdigest()

    (tmp_path / "lio_map.pcd").write_bytes(b"changed")
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        builder.build_alignment(_args(bundle, base_yaw_deg=30.0, lio_map=str(meta)))


def test_builder_cli_writes_loadable_alignment(tmp_path):
    out = tmp_path / "alignment.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/build_prior_lio_alignment.py"), str(_bundle(tmp_path)),
                    "--base-xy", "1,2", "--base-yaw-deg", "30", "--output", str(out)], check=True,
                   capture_output=True, text=True)
    loaded = PriorMapAlignment.load(out)
    assert loaded.can_rank_geometry() is True
    assert loaded.prior_to_runtime((1.0, 2.0))[:2] == pytest.approx((-0.2, 0.0), abs=1e-9)


def _stub_env(tmp_path: Path, has_session: int) -> dict:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "tmux.log"
    # tmux 桩：记录调用；has-session 返回值可控；send-keys 模拟 Point-LIO 退出时写图。
    (bin_dir / "tmux").write_text(f"""#!/usr/bin/env bash
echo "$*" >> '{log}'
case "$1" in
  has-session) exit {has_session} ;;
  display-message) echo 1700000000 ;;
  send-keys) printf 'pcd-bytes' > "$POINT_LIO_PCD" ;;
esac
""")
    (bin_dir / "tmux").chmod(0o755)
    for name in ("ros.bash", "livox.bash", "point_lio.bash", "config.yaml"):
        (tmp_path / name).write_text("")
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}",
               ROS_SETUP_BASH=str(tmp_path / "ros.bash"), LIVOX_SETUP_BASH=str(tmp_path / "livox.bash"),
               POINT_LIO_SETUP_BASH=str(tmp_path / "point_lio.bash"), POINT_LIO_CONFIG=str(tmp_path / "config.yaml"),
               POINT_LIO_PCD=str(tmp_path / "scans.pcd"))
    return env


def _run(script: str, env: dict, *args):
    return subprocess.run(["bash", str(ROOT / "scripts" / script), *args], env=env, capture_output=True, text=True)


def test_start_lio_rejects_bad_location_config_before_killing_session(tmp_path):
    env = _stub_env(tmp_path, has_session=1)
    log = tmp_path / "tmux.log"

    result = _run("start_orin_lio_for_strive.sh", dict(env, LIO_LOCATION_MODE="1", LIO_MAP_PATH="rel.pcd"))
    assert result.returncode == 3
    pcd = tmp_path / "map.pcd"
    pcd.write_text("x")
    result = _run("start_orin_lio_for_strive.sh",
                  dict(env, LIO_LOCATION_MODE="1", LIO_MAP_PATH=str(pcd), LIO_PCD_SAVE_EN="1"))
    assert result.returncode == 3
    assert not log.exists()


def test_start_lio_passes_location_and_save_params(tmp_path):
    env = _stub_env(tmp_path, has_session=1)
    pcd = tmp_path / "map.pcd"
    pcd.write_text("x")

    assert _run("start_orin_lio_for_strive.sh", dict(env, LIO_LOCATION_MODE="1", LIO_MAP_PATH=str(pcd))).returncode == 0
    calls = (tmp_path / "tmux.log").read_text()
    assert calls.index("kill-session") < calls.index("new-session")
    assert "location_mode:=true" in calls
    assert f"map_path:={pcd}" in calls
    assert "pcd_save.pcd_save_en:=false" in calls

    (tmp_path / "tmux.log").unlink()
    assert _run("start_orin_lio_for_strive.sh", dict(env, LIO_PCD_SAVE_EN="1")).returncode == 0
    calls = (tmp_path / "tmux.log").read_text()
    assert "pcd_save.pcd_save_en:=true" in calls
    assert "location_mode:=false" in calls
    assert "map_path:=" not in calls


def test_save_lio_map_waits_for_new_pcd_and_closes_session(tmp_path):
    env = _stub_env(tmp_path, has_session=0)
    out = tmp_path / "saved"

    result = _run("save_orin_lio_map.sh", dict(env, SAVE_TIMEOUT_S="20"), str(out))
    assert result.returncode == 0, result.stderr
    meta = json.loads((out / "lio_map.json").read_text())
    assert (out / "lio_map.pcd").read_bytes() == b"pcd-bytes"
    assert meta["sha256"] == hashlib.sha256(b"pcd-bytes").hexdigest()
    assert meta["frame_id"] == "camera_init"
    assert meta["mapping_session_created_unix"] == 1700000000
    calls = (tmp_path / "tmux.log").read_text()
    assert calls.index("send-keys") < calls.index("kill-session")

    assert _run("save_orin_lio_map.sh", env, str(out)).returncode == 2
