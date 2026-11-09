from __future__ import annotations

import hashlib
import json
import os
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from verification.images import image_input
from dataclasses import asdict, dataclass, field
from typing import Any

from llm_utils.cognav_llm_adapter import get_client_and_model
from prompting.registry import RELATION_VERIFY
from prompting.schemas import HAS_PYDANTIC, ParsedRelationResult
from prompting.templates import RELATION_VERIFIER_PROMPT

from .semantic_edges import DynamicSemanticEdgeVerifier, SemanticEdge, relation_view_is_bound


@dataclass
class RelationVerificationRequest:
    relation: str
    subject: dict[str, Any]
    object_: dict[str, Any]
    evidence_views: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["object"] = data.pop("object_")
        return data


class VLMRelationVerifier:
    """Prompt-first relation verifier used after geometric prefiltering."""

    def __init__(self, vlm: str = ""):
        self.vlm = vlm

    def __call__(
        self,
        relation: str,
        subject: dict[str, Any],
        object_: dict[str, Any],
        evidence_views: list[dict[str, Any]],
        prompt: str,
    ) -> dict[str, Any]:
        if os.getenv("STRIVE_RELATION_VERIFIER", "auto").lower() in ("0", "false", "no", "off"):
            return {"verified": False, "confidence": 0.0, "need_better_view": True, "reason": "relation verifier disabled"}
        if os.getenv("LLM_OFFLINE", "0").lower() in ("1", "true", "yes", "on"):
            return {"verified": False, "confidence": 0.0, "need_better_view": True, "reason": "llm offline"}
        if not HAS_PYDANTIC:
            return {"verified": False, "confidence": 0.0, "need_better_view": True, "reason": "pydantic unavailable"}

        views, images = [], []
        for view in evidence_views:
            if not relation_view_is_bound(view, subject["id"], object_["id"]):
                continue
            path = view.get("rgb_path") or view.get("image_ref") or view.get("path")
            block, archived, digest = image_input(path, Path(path).parent / "relation_evidence")
            bound = {**view, "rgb_path": archived, "image_sha256": digest, "regions": []}
            region_blocks = []
            with Image.open(archived) as frame:
                for role, obj in (("subject", subject), ("reference", object_)):
                    box = list(view["object_boxes"][obj["id"]])
                    crop = frame.crop(box).convert("RGB")
                    buffer = BytesIO()
                    crop.save(buffer, format="PNG")
                    source_crop_digest = hashlib.sha256(buffer.getvalue()).hexdigest()
                    width = max(360, crop.width)
                    canvas = Image.new("RGB", (width, crop.height + 80), "white")
                    title = Image.new("RGB", (120, 16), "white")
                    ImageDraw.Draw(title).text((2, 2), role.upper(), fill="black")
                    canvas.paste(title.resize((360, 48), resample=Image.NEAREST), (0, 0))
                    draw = ImageDraw.Draw(canvas)
                    draw.text((6, 48), "Frame " + view["id"], fill="black")
                    draw.text((6, 62), obj["id"], fill="black")
                    x, y = (width - crop.width) // 2, 80
                    canvas.paste(crop, (x, y))
                    buffer = BytesIO()
                    canvas.save(buffer, format="PNG")
                    data = buffer.getvalue()
                    crop_path = Path(archived).parent / (hashlib.sha256(data).hexdigest() + ".png")
                    if not crop_path.exists():
                        crop_path.write_bytes(data)
                    crop_block, crop_archive, crop_digest = image_input(str(crop_path), crop_path.parent)
                    region = {"role": role, "instance_id": obj["id"],
                              "requested_concept": obj["requested_concept"], "demand": obj.get("demand", {}),
                              "attributes": obj.get("attributes", {}),
                              "source_frame_id": view["id"], "source_image_sha256": digest,
                              "bbox_xyxy_in_source": box, "rgb_path": crop_archive,
                              "image_sha256": crop_digest,
                              "presentation": {"source_crop_sha256": source_crop_digest,
                                               "unscaled_crop_box_in_image": [x, y, x + crop.width, y + crop.height],
                                               "outside_crop_role_caption": role}}
                    bound["regions"].append(region)
                    region_blocks.extend([{"type": "text", "text": json.dumps(region, ensure_ascii=False)}, crop_block])
            views.append(bound)
            images.extend([{"type": "text", "text": json.dumps(bound, ensure_ascii=False)}, block, *region_blocks])
        if not views:
            return {"verified": False, "confidence": 0.0, "need_better_view": True,
                    "reason": "No readable frame with both instance regions bound to it."}
        request = RelationVerificationRequest(relation, subject, object_, views)
        content = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": json.dumps(request.as_dict(), ensure_ascii=False, indent=2)},
            *images,
        ]
        client, model = get_client_and_model(self.vlm)
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": RELATION_VERIFIER_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format=ParsedRelationResult,
            trace_label=RELATION_VERIFY.trace_label,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            return {"verified": False, "confidence": 0.0, "need_better_view": True,
                    "reason": "Relation verifier returned no parsed judgment."}
        identities = (parsed.subject_identity.verdict, parsed.reference_identity.verdict)
        if "false" in identities:
            verdict = "false"
        elif identities == ("true", "true"):
            verdict = parsed.relation.verdict
        else:
            verdict = "unknown"
        judgments = (("subject_identity", parsed.subject_identity),
                     ("reference_identity", parsed.reference_identity), ("relation", parsed.relation))
        return {
            "verified": verdict == "true",
            "confidence": float(parsed.confidence),
            "need_better_view": verdict == "unknown",
            "identity_judgments": {
                role: {"verdict": judgment.verdict, "reason": judgment.reason,
                       "instance_id": obj["id"], "requested_concept": obj["requested_concept"],
                       "demand": obj.get("demand", {}),
                       "source_crop_sha256": sorted({region["presentation"]["source_crop_sha256"]
                           for view in views for region in view["regions"] if region["role"] == role})}
                for role, obj, judgment in (("subject", subject, parsed.subject_identity),
                                            ("reference", object_, parsed.reference_identity))
            },
            "reason": "; ".join(f"{role}={judgment.verdict}: {judgment.reason}"
                                for role, judgment in judgments),
        }


class DynamicRelationService:
    """Geometry-first, VLM-on-demand relation service."""

    def __init__(self, vlm: str = ""):
        self.edge_verifier = DynamicSemanticEdgeVerifier()
        self.vlm_callback = VLMRelationVerifier(vlm=vlm)

    def reset(self):
        self.edge_verifier = DynamicSemanticEdgeVerifier()

    def verify(
        self,
        *,
        subject: dict[str, Any],
        relation: str,
        object_: dict[str, Any],
        evidence_views: list[dict[str, Any]],
        use_vlm: bool = True,
    ) -> SemanticEdge:
        callback = self.vlm_callback if use_vlm else None
        return self.edge_verifier.verify(
            subject=subject,
            relation=relation,
            object_=object_,
            evidence_views=evidence_views,
            vlm_callback=callback,
        )

    def as_dict(self) -> dict[str, Any]:
        return self.edge_verifier.cache.as_dict()
