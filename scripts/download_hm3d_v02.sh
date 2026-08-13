#!/usr/bin/env bash
# Download HM3D v0.2 scene assets from the official public Matterport links.
set -euo pipefail

STRIVE_ROOT="${STRIVE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STRIVE_DATA_ROOT="${STRIVE_DATA_ROOT:-$STRIVE_ROOT/data}"
SPLITS="${HM3D_SPLITS:-val minival}"
DOWNLOAD_DIR="${HM3D_DOWNLOAD_DIR:-$STRIVE_DATA_ROOT/downloads/hm3d_v0.2}"
TARGET_DIR="$STRIVE_DATA_ROOT/scene_datasets/hm3d_v0.2"

BASE_URL="https://api.matterport.com/resources/habitat"
KINDS="${HM3D_ASSET_KINDS:-glb habitat semantic-annots semantic-configs}"

download_one() {
  local split="$1"
  local kind="$2"
  local out="$DOWNLOAD_DIR/hm3d-${split}-${kind}-v0.2.tar"
  local url="$BASE_URL/hm3d-${split}-${kind}-v0.2.tar"
  if [[ -f "$out" ]]; then
    echo "[hm3d] exists: $out"
    return
  fi
  echo "[hm3d] download: $url"
  if command -v curl >/dev/null 2>&1; then
    curl -L --continue-at - -o "$out" "$url"
  else
    wget -c -O "$out" "$url"
  fi
}

extract_one() {
  local archive="$1"
  local tmp
  tmp="$(mktemp -d)"
  echo "[hm3d] extract: $archive"
  tar -xf "$archive" -C "$tmp"

  mkdir -p "$TARGET_DIR"
  if [[ -d "$tmp/hm3d_v0.2" ]]; then
    cp -a "$tmp/hm3d_v0.2/." "$TARGET_DIR/"
  elif [[ -d "$tmp/hm3d" ]]; then
    cp -a "$tmp/hm3d/." "$TARGET_DIR/"
  else
    cp -a "$tmp/." "$TARGET_DIR/"
  fi
  rm -rf "$tmp"
}

mkdir -p "$DOWNLOAD_DIR" "$TARGET_DIR"

for split in $SPLITS; do
  for kind in $KINDS; do
    download_one "$split" "$kind"
  done
done

shopt -s nullglob
archives=("$DOWNLOAD_DIR"/*.tar)
if [[ "${#archives[@]}" -eq 0 ]]; then
  echo "[hm3d] no archives found under $DOWNLOAD_DIR" >&2
  exit 2
fi

for archive in "${archives[@]}"; do
  extract_one "$archive"
done

echo "[hm3d] target: $TARGET_DIR"
test -f "$TARGET_DIR/hm3d_annotated_basis.scene_dataset_config.json"
find "$TARGET_DIR" -maxdepth 3 -name "*.basis.glb" | head
find "$TARGET_DIR" -maxdepth 3 -name "*.basis.navmesh" | head
find "$TARGET_DIR" -maxdepth 3 -name "*.semantic.glb" | head
echo "[hm3d] OK"
