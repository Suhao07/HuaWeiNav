"""Preserve the exact image bytes supplied to a verification request."""
import base64
from hashlib import sha256
from mimetypes import guess_type
from pathlib import Path


def image_input(path: str, evidence_dir: Path) -> tuple[dict, str, str]:
    """Read once, archive by content hash, and encode the same bytes for the model.

    Args:
        path: Readable image produced by the perception backend.
        evidence_dir: Task-owned directory for immutable evidence blobs.

    Returns:
        Model image block, absolute archived reference, and SHA-256 digest.

    Raises:
        OSError: The source or evidence directory is unavailable.
        ValueError: The file extension does not identify a supported image type.

    核心关联：evidence_id = SHA256(image_bytes)。请求与归档复用同一次读取，
    从而避免路径复用或采集线程覆盖文件造成证据与模型输入不一致。
    """
    source = Path(path)
    mime = guess_type(source.name)[0]
    if mime not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise ValueError(f"Unsupported verifier image type: {source.suffix}")
    data = source.read_bytes()
    digest = sha256(data).hexdigest()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    archived = (evidence_dir / (digest + source.suffix.lower())).resolve()
    if not archived.exists():
        archived.write_bytes(data)
    block = {"type": "image_url", "image_url": {
        "url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"}}
    return block, str(archived), digest


def image_dependencies_valid(dependencies: dict[str, str]) -> bool:
    """Check stored image fingerprints before reusing their judgments.

    Missing archives are expired evidence. Other IO errors retain their causes.
    """
    for key, expected in dependencies.items():
        if key.startswith("image:"):
            try:
                data = Path(key[len("image:"):]).read_bytes()
            except FileNotFoundError:
                return False
            if sha256(data).hexdigest() != expected:
                return False
    return True
