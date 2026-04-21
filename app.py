from __future__ import annotations

import io
import json
import os
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any


def _bootstrap_site_packages() -> None:
    candidates: list[Path] = []
    explicit_path = os.environ.get("MEDICALAI_EXTRA_SITE_PACKAGES")
    if explicit_path:
        candidates.append(Path(explicit_path))

    executable_path = Path(sys.executable).resolve()
    for parent in executable_path.parents:
        if parent.name == "envs":
            candidates.append(parent.parent / "Lib" / "site-packages")

    candidates.extend(
        [
            Path.home() / "anaconda3" / "Lib" / "site-packages",
            Path.home() / "miniconda3" / "Lib" / "site-packages",
        ]
    )

    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.append(candidate_str)


try:
    from flask import (
        Flask,
        flash,
        g,
        redirect,
        render_template,
        request,
        send_file,
        session,
        url_for,
    )
except ModuleNotFoundError:
    _bootstrap_site_packages()
    from flask import (
        Flask,
        flash,
        g,
        redirect,
        render_template,
        request,
        send_file,
        session,
        url_for,
    )

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
    from werkzeug.exceptions import RequestEntityTooLarge
    from werkzeug.security import check_password_hash, generate_password_hash
    from werkzeug.utils import secure_filename
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "MedicalAI requires Flask, NumPy, Pillow, and PyTorch in the active Python environment."
    ) from exc

from model import AtriumUNet, BinaryClassifier
from nifti_utils import extract_middle_slice, load_nifti_volume

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"
UPLOAD_DIR = STATIC_DIR / "uploads"
GENERATED_DIR = STATIC_DIR / "generated"
DATABASE_PATH = BASE_DIR / "medicalai.db"

INPUT_SIZE = (128, 128)
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
XRAY_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MRI_EXTENSIONS = {".nii", ".nii.gz"}
DEFAULT_THRESHOLD = 0.50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESAMPLING = Image.Resampling if hasattr(Image, "Resampling") else Image

DEFAULT_ADMIN_USERNAME = os.environ.get("MEDICALAI_ADMIN_USER", "admin")
DEFAULT_ADMIN_PASSWORD = os.environ.get("MEDICALAI_ADMIN_PASSWORD", "MedicalAI2026")
DEFAULT_ADMIN_NAME = os.environ.get("MEDICALAI_ADMIN_NAME", "Project Administrator")

PROJECT_SHORT_TITLE = "Automated Medical Image Analysis"
PROJECT_SUBTITLE = "Disease Diagnosis Using Deep Learning"
PROJECT_FULL_TITLE = "Automated Medical Image Analysis for Disease Diagnosis Using Deep Learning"
COLLEGE_NAME = "Mahatma Gandhi Mission's College Of Engineering & Technology, Noida, NCR"
DEPARTMENT_NAME = "Department of Computer Science Engineering"
GUIDE_NAME = "Mrs. Pooja Singh"
COORDINATOR_NAME = "Mr. Abhishek Chaudhary"
ACADEMIC_YEAR = "Academic Year 2022-2026"
TEAM_MEMBERS = [
    {"name": "Kanish Shandilya", "role": "Frontend, system integration, project deployment"},
    {"name": "Devansh Dhingra", "role": "Model training, experimentation, dataset preparation"},
    {"name": "Komal Kashyap", "role": "Research support, documentation, validation assistance"},
]

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ModelCard:
    key: str
    title: str
    modality: str
    analysis_type: str
    accepted_formats: str
    accept_attr: str
    input_guidance: str
    summary: str
    output_guidance: str
    accent: str
    badge: str
    sample_relative: str | None = None


MODEL_CARDS = [
    ModelCard(
        key="cardiac",
        title="Cardiac Detection",
        modality="Chest X-ray",
        analysis_type="Binary classification",
        accepted_formats="PNG, JPG or JPEG chest X-ray image",
        accept_attr=".png,.jpg,.jpeg",
        input_guidance="Upload a frontal chest radiograph with the thorax centered. This module accepts exported chest X-ray image files in PNG or JPG format.",
        summary="Screens a chest X-ray for heart-related radiographic patterns and returns a confidence score plus activation heatmap.",
        output_guidance="Generates a screening score and an activation map showing which chest regions most influenced the model.",
        accent="#b7543c",
        badge="X-ray",
        sample_relative="uploads/download.jpeg",
    ),
    ModelCard(
        key="pneumonia",
        title="Pneumonia Detection",
        modality="Chest X-ray",
        analysis_type="Binary classification",
        accepted_formats="PNG, JPG or JPEG chest X-ray image",
        accept_attr=".png,.jpg,.jpeg",
        input_guidance="Upload a chest X-ray image in PNG or JPG format. The model was trained on pneumonia-labelled chest radiographs and is presented here as an AI-assisted screening workflow.",
        summary="Evaluates lung opacity patterns associated with pneumonia and returns a risk-oriented score plus activation heatmap.",
        output_guidance="Generates a screening score and highlights the radiographic zones that most influenced the prediction.",
        accent="#4d7c42",
        badge="X-ray",
        sample_relative="uploads/download.jpeg",
    ),
    ModelCard(
        key="atrium",
        title="Atrium Segmentation",
        modality="Cardiac MRI",
        analysis_type="Segmentation",
        accepted_formats="NIfTI volume in .nii or .nii.gz format",
        accept_attr=".nii,.nii.gz",
        input_guidance="Upload a cardiac MRI volume in NIfTI format (.nii or .nii.gz). NIfTI is a common research format for MRI volumes. The platform extracts the middle slice, normalizes it, and predicts the atrial region as a binary mask.",
        summary="Segments the atrial region from a cardiac MRI volume and returns the original slice, predicted mask, and overlay.",
        output_guidance="Generates a binary segmentation mask and a color overlay over the extracted MRI slice.",
        accent="#0f766e",
        badge="MRI",
        sample_relative=None,
    ),
]
MODEL_LOOKUP = {card.key: card for card in MODEL_CARDS}


class ModelRegistry:
    def __init__(self, device: torch.device):
        self.device = device
        self._loaded = False
        self._load_error: str | None = None
        self.atrium: AtriumUNet | None = None
        self.cardiac: BinaryClassifier | None = None
        self.pneumonia: BinaryClassifier | None = None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> None:
        if self._loaded or self._load_error:
            return

        try:
            self.atrium = AtriumUNet().to(self.device)
            self.atrium.load_state_dict(_load_state_dict(BASE_DIR / "atrium_model.pth", self.device))
            self.atrium.eval()

            self.cardiac = BinaryClassifier().to(self.device)
            self.cardiac.load_state_dict(_load_state_dict(BASE_DIR / "heart_model.pth", self.device))
            self.cardiac.eval()

            self.pneumonia = BinaryClassifier().to(self.device)
            self.pneumonia.load_state_dict(
                _load_state_dict(BASE_DIR / "pneumonia_model.pth", self.device)
            )
            self.pneumonia.eval()
            self._loaded = True
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"

    def get(self, model_key: str) -> AtriumUNet | BinaryClassifier:
        self.load()
        if self._load_error:
            raise RuntimeError(self._load_error)

        model = getattr(self, model_key, None)
        if model is None:
            raise KeyError(f"Unknown model requested: {model_key}")
        return model


def _load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _normalize_map(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - minimum) / (maximum - minimum)


def _format_timestamp(dt: datetime | None = None) -> str:
    dt = dt or datetime.now(timezone.utc)
    return dt.astimezone().strftime("%d %b %Y, %I:%M %p")


def _compound_extension(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(lower).suffix


def _allowed_extension(model_key: str, filename: str) -> bool:
    extension = _compound_extension(filename)
    if model_key == "atrium":
        return extension in MRI_EXTENSIONS
    return extension in XRAY_EXTENSIONS


def _prepare_xray_tensor(image: Image.Image, *, minus_one_to_one: bool = False) -> torch.Tensor:
    grayscale = ImageOps.grayscale(image)
    resized = grayscale.resize(INPUT_SIZE, RESAMPLING.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    if minus_one_to_one:
        array = (array - 0.5) / 0.5
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(DEVICE)


def _prepare_mri_tensor(slice_array: np.ndarray) -> torch.Tensor:
    normalized = _normalize_map(slice_array)
    image = Image.fromarray(np.clip(normalized * 255.0, 0, 255).astype(np.uint8), mode="L")
    resized = image.resize(INPUT_SIZE, RESAMPLING.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(DEVICE)


def _save_generated_image(image: Image.Image, stem: str, suffix: str) -> str:
    filename = f"{stem}_{suffix}.png"
    image.save(GENERATED_DIR / filename)
    return f"generated/{filename}"


def _save_uploaded_xray(file_storage: Any) -> tuple[str, Image.Image]:
    original_name = secure_filename(file_storage.filename or "")
    if not original_name or not _allowed_extension("cardiac", original_name):
        raise ValueError("Please upload a chest X-ray image in PNG, JPG, or JPEG format.")

    try:
        image = Image.open(file_storage.stream)
        image = ImageOps.exif_transpose(image).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded X-ray file could not be read as an image.") from exc

    saved_name = f"{uuid.uuid4().hex}.png"
    image.save(UPLOAD_DIR / saved_name, format="PNG")
    return saved_name, image


def _save_uploaded_mri(file_storage: Any) -> tuple[str, Path]:
    original_name = secure_filename(file_storage.filename or "")
    if not original_name or not _allowed_extension("atrium", original_name):
        raise ValueError("Atrium segmentation only accepts cardiac MRI volumes in .nii or .nii.gz format.")

    extension = ".nii.gz" if original_name.lower().endswith(".nii.gz") else ".nii"
    saved_name = f"{uuid.uuid4().hex}{extension}"
    save_path = UPLOAD_DIR / saved_name
    file_storage.save(save_path)
    return saved_name, save_path


def _render_segmentation_assets(
    grayscale_image: Image.Image, binary_mask: np.ndarray
) -> tuple[Image.Image, Image.Image]:
    mask_image = Image.fromarray((binary_mask * 255).astype(np.uint8), mode="L")
    softened = mask_image.filter(ImageFilter.GaussianBlur(radius=1.8))

    grayscale_base = grayscale_image.convert("RGBA")
    color_layer = Image.new("RGBA", grayscale_image.size, (20, 188, 196, 0))
    color_layer.putalpha(softened.point(lambda pixel: int(pixel * 0.72)))
    overlay = Image.alpha_composite(grayscale_base, color_layer)

    outline = mask_image.filter(ImageFilter.FIND_EDGES).point(lambda pixel: 255 if pixel > 8 else 0)
    outline_layer = Image.new("RGBA", grayscale_image.size, (8, 96, 168, 0))
    outline_layer.putalpha(outline)
    overlay = Image.alpha_composite(overlay, outline_layer)

    mask_preview = ImageOps.colorize(mask_image, black="#0f172a", white="#24d7f2").convert("RGB")
    return overlay.convert("RGB"), mask_preview


def _render_heatmap_overlay(image: Image.Image, activation_map: np.ndarray) -> Image.Image:
    grayscale = np.asarray(ImageOps.grayscale(image).convert("RGB"), dtype=np.float32)
    activation = _normalize_map(activation_map)
    activation_u8 = np.clip(activation * 255.0, 0, 255).astype(np.uint8)

    heatmap = np.zeros((activation_u8.shape[0], activation_u8.shape[1], 3), dtype=np.float32)
    heatmap[..., 0] = np.clip(activation_u8 * 1.18, 0, 255)
    heatmap[..., 1] = np.clip(np.maximum(activation_u8 - 28, 0) * 0.82, 0, 255)
    heatmap[..., 2] = np.clip(np.maximum(activation_u8 - 120, 0) * 0.26, 0, 255)

    alpha = (activation[..., None] * 0.62).astype(np.float32)
    overlay = (grayscale * (1.0 - alpha) + heatmap * alpha).clip(0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def _build_classifier_result(
    *,
    model_key: str,
    uploaded_name: str,
    image: Image.Image,
    score: float,
    activation: np.ndarray,
) -> dict[str, Any]:
    card = MODEL_LOOKUP[model_key]
    is_positive = score >= DEFAULT_THRESHOLD
    confidence = max(score, 1.0 - score) * 100.0
    score_percent = score * 100.0

    heatmap = _render_heatmap_overlay(image, activation)
    hero_path = _save_generated_image(heatmap, Path(uploaded_name).stem, f"{model_key}_heatmap")

    if model_key == "cardiac":
        positive_class_label = "Heart-related radiographic screening findings"
        plain_meaning = (
            "The model found a stronger match to its positive cardiac screening examples than to its negative examples."
            if is_positive
            else "The model found a stronger match to its negative cardiac screening examples than to its positive examples."
        )
        significance = (
            "This workflow reviews heart-related radiographic patterns such as changes in the cardiac silhouette and surrounding chest appearance."
        )
        caution = (
            "This is an AI-assisted screening result only. It should not be treated as confirmation of a specific cardiac disease without clinical review."
        )
    else:
        positive_class_label = "Pneumonia-like radiographic screening findings"
        plain_meaning = (
            "The model found a stronger match to pneumonia-like chest X-ray examples than to non-pneumonia examples."
            if is_positive
            else "The model found a stronger match to non-pneumonia chest X-ray examples than to pneumonia-like examples."
        )
        significance = (
            "This workflow reviews lung opacity patterns that can be associated with pneumonia on a chest X-ray."
        )
        caution = (
            "This is an AI-assisted screening result only. Infection status still needs clinical context, examination, and radiology review."
        )

    verdict = "High positive screening signal" if is_positive else "Low positive screening signal"
    summary = (
        f"The {card.title.lower()} model assigned {score_percent:.1f}% AI screening confidence to the positive class. "
        f"With a 50% decision threshold, this run produced a {'high' if is_positive else 'low'} positive screening signal."
    )

    return {
        "model_key": model_key,
        "model_title": card.title,
        "title": card.title,
        "modality": card.modality,
        "analysis_type": card.analysis_type,
        "verdict": verdict,
        "status_class": "is-positive" if is_positive else "is-negative",
        "score_label": "AI screening confidence",
        "score_display": f"{score_percent:.1f}%",
        "meter_value": round(score_percent, 1),
        "summary": summary,
        "plain_meaning": plain_meaning,
        "clinical_significance": f"{significance} {caution}",
        "technical_note": "The heatmap highlights image regions that influenced the classifier most strongly. It is an attention-style visual, not a pixel-perfect lesion boundary.",
        "hero_artifact": hero_path,
        "artifacts": [
            {"label": "Uploaded chest X-ray", "path": f"uploads/{uploaded_name}"},
            {"label": "Model activation heatmap", "path": hero_path},
        ],
        "details": [
            {"label": "Accepted site input", "value": card.accepted_formats},
            {"label": "Positive signal refers to", "value": positive_class_label},
            {"label": "Screening outcome", "value": verdict},
            {"label": "Decision threshold", "value": "50%"},
            {"label": "Confidence around selected outcome", "value": f"{confidence:.1f}%"},
            {"label": "Inference canvas", "value": "128 x 128 grayscale"},
        ],
        "report_notes": [
            f"Modality: {card.modality}",
            f"AI screening confidence: {score_percent:.1f}%",
            f"Verdict: {verdict}",
            f"Plain-language meaning: {plain_meaning}",
            f"Clinical significance: {significance}",
            f"Clinical caution: {caution}",
        ],
        "cleanup_paths": [
            f"uploads/{uploaded_name}",
            hero_path,
        ],
    }


def _build_atrium_result(
    *,
    uploaded_name: str,
    volume_shape: tuple[int, ...],
    slice_image: Image.Image,
    binary_mask: np.ndarray,
) -> dict[str, Any]:
    card = MODEL_LOOKUP["atrium"]
    overlay, mask_preview = _render_segmentation_assets(slice_image.convert("RGBA"), binary_mask)
    stem = Path(uploaded_name).stem
    slice_path = _save_generated_image(slice_image.convert("RGB"), stem, "mri_slice")
    overlay_path = _save_generated_image(overlay, stem, "atrium_overlay")
    mask_path = _save_generated_image(mask_preview, stem, "atrium_mask")
    coverage = float(binary_mask.mean() * 100.0)

    return {
        "model_key": "atrium",
        "model_title": card.title,
        "title": card.title,
        "modality": card.modality,
        "analysis_type": card.analysis_type,
        "verdict": "Segmentation generated",
        "status_class": "is-info",
        "score_label": "Mask coverage",
        "score_display": f"{coverage:.1f}%",
        "meter_value": round(min(coverage, 100.0), 1),
        "summary": "The uploaded MRI volume was converted to a middle slice and segmented by the atrium model.",
        "plain_meaning": "The colored overlay shows the region the model believes corresponds to the atrial structure on the extracted cardiac MRI slice.",
        "clinical_significance": "Segmentation is useful for anatomical localization and visual review. It should be treated as an assistive output, not as a definitive diagnosis.",
        "technical_note": "The current build extracts the middle MRI slice from the uploaded NIfTI volume before running the U-Net style segmentation model.",
        "hero_artifact": overlay_path,
        "artifacts": [
            {"label": "Extracted MRI slice", "path": slice_path},
            {"label": "Atrium overlay", "path": overlay_path},
            {"label": "Predicted binary mask", "path": mask_path},
        ],
        "details": [
            {"label": "Accepted site input", "value": card.accepted_formats},
            {"label": "Uploaded volume shape", "value": " x ".join(str(v) for v in volume_shape)},
            {"label": "Segmentation threshold", "value": "50%"},
            {"label": "Inference canvas", "value": "128 x 128 grayscale"},
        ],
        "report_notes": [
            f"Modality: {card.modality}",
            f"Volume shape: {' x '.join(str(v) for v in volume_shape)}",
            f"Mask coverage: {coverage:.1f}%",
            "Plain-language meaning: The predicted mask highlights the atrial region on the extracted MRI slice.",
            "Clinical caution: The output is anatomical guidance and should be reviewed with source imaging context.",
        ],
        "cleanup_paths": [
            f"uploads/{uploaded_name}",
            slice_path,
            overlay_path,
            mask_path,
        ],
    }


def _run_classifier(model_key: str, uploaded_name: str, image: Image.Image) -> dict[str, Any]:
    minus_one_to_one = model_key == "pneumonia"
    tensor = _prepare_xray_tensor(image, minus_one_to_one=minus_one_to_one)
    model = MODELS.get(model_key)

    model.zero_grad()
    features = model.forward_features(tensor)
    features.retain_grad()
    logits = model.forward_from_features(features)
    score = float(torch.sigmoid(logits).item())
    logits.backward(torch.ones_like(logits))

    gradients = features.grad.detach()
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * features.detach()).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(image.height, image.width), mode="bilinear", align_corners=False)
    activation = _normalize_map(cam.squeeze().detach().cpu().numpy())

    return _build_classifier_result(
        model_key=model_key,
        uploaded_name=uploaded_name,
        image=image,
        score=score,
        activation=activation,
    )


def _run_atrium(uploaded_name: str, upload_path: Path) -> dict[str, Any]:
    volume = load_nifti_volume(upload_path)
    slice_array = extract_middle_slice(volume)
    normalized_slice = _normalize_map(slice_array)
    slice_image = Image.fromarray(
        np.clip(normalized_slice * 255.0, 0, 255).astype(np.uint8), mode="L"
    ).resize((slice_array.shape[1], slice_array.shape[0]), RESAMPLING.NEAREST)

    tensor = _prepare_mri_tensor(slice_array)
    model = MODELS.get("atrium")

    with torch.no_grad():
        prediction = torch.sigmoid(model(tensor))

    mask = prediction.squeeze().detach().cpu().numpy()
    binary_small = (_normalize_map(mask) >= DEFAULT_THRESHOLD).astype(np.uint8)
    binary_mask = np.asarray(
        Image.fromarray((binary_small * 255).astype(np.uint8), mode="L").resize(
            slice_image.size, RESAMPLING.NEAREST
        )
    )
    binary_mask = (binary_mask > 127).astype(np.uint8)

    return _build_atrium_result(
        uploaded_name=uploaded_name,
        volume_shape=tuple(int(dim) for dim in volume.shape),
        slice_image=slice_image,
        binary_mask=binary_mask,
    )


def _build_report_text(record: dict[str, Any]) -> str:
    result = record["result"]
    lines = [
        "MedicalAI Imaging Platform Report",
        "=" * 32,
        "",
        f"Analysis ID: {record['id']}",
        f"Date: {record['created_at']}",
        f"User: {record['user_name']}",
        f"Model: {result['model_title']}",
        f"Modality: {result['modality']}",
        f"Verdict: {result['verdict']}",
        f"{result['score_label']}: {result['score_display']}",
        "",
        "Summary",
        result["summary"],
        "",
        "What This Means",
        result["plain_meaning"],
        "",
        "Clinical Significance",
        result["clinical_significance"],
        "",
        "Technical Note",
        result["technical_note"],
    ]
    if result.get("report_notes"):
        lines.extend(["", "Additional Notes"])
        lines.extend(result["report_notes"])
    return "\n".join(lines)


def _get_db() -> sqlite3.Connection:
    connection = getattr(g, "_database", None)
    if connection is None:
        connection = sqlite3.connect(DATABASE_PATH)
        connection.row_factory = sqlite3.Row
        g._database = connection
    return connection


def init_database() -> None:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                model_key TEXT NOT NULL,
                title TEXT NOT NULL,
                verdict TEXT NOT NULL,
                score_display TEXT NOT NULL,
                created_at TEXT NOT NULL,
                cover_artifact TEXT,
                result_json TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            """
        )

        existing = connection.execute(
            "SELECT id FROM users WHERE username = ?",
            (DEFAULT_ADMIN_USERNAME,),
        ).fetchone()
        if existing is None:
            connection.execute(
                """
                INSERT INTO users (username, password_hash, display_name, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    DEFAULT_ADMIN_USERNAME,
                    generate_password_hash(DEFAULT_ADMIN_PASSWORD),
                    DEFAULT_ADMIN_NAME,
                    _format_timestamp(),
                ),
            )
        connection.commit()
    finally:
        connection.close()


def _save_analysis(user_id: int, result: dict[str, Any]) -> int:
    connection = _get_db()
    cursor = connection.execute(
        """
        INSERT INTO analyses (user_id, model_key, title, verdict, score_display, created_at, cover_artifact, result_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            result["model_key"],
            result["title"],
            result["verdict"],
            result["score_display"],
            _format_timestamp(),
            result.get("hero_artifact"),
            json.dumps(result),
        ),
    )
    connection.commit()
    return int(cursor.lastrowid)


def _fetch_user(user_id: int) -> sqlite3.Row | None:
    return _get_db().execute(
        "SELECT id, username, display_name FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()


def _fetch_history(user_id: int, *, limit: int = 8) -> list[dict[str, Any]]:
    rows = _get_db().execute(
        """
        SELECT id, model_key, title, verdict, score_display, created_at, cover_artifact
        FROM analyses
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (user_id, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def _fetch_analysis(user_id: int, analysis_id: int) -> dict[str, Any] | None:
    row = _get_db().execute(
        """
        SELECT a.id, a.created_at, a.result_json, u.display_name
        FROM analyses a
        JOIN users u ON u.id = a.user_id
        WHERE a.user_id = ? AND a.id = ?
        """,
        (user_id, analysis_id),
    ).fetchone()
    if row is None:
        return None
    return {
        "id": int(row["id"]),
        "created_at": row["created_at"],
        "user_name": row["display_name"],
        "result": json.loads(row["result_json"]),
    }


def _safe_remove_static_file(relative_path: str) -> None:
    target = (STATIC_DIR / relative_path).resolve()
    try:
        target.relative_to(STATIC_DIR.resolve())
    except ValueError:
        return
    if target.exists() and target.is_file():
        target.unlink()


def _delete_analysis_record(user_id: int, analysis_id: int) -> bool:
    row = _get_db().execute(
        "SELECT result_json FROM analyses WHERE id = ? AND user_id = ?",
        (analysis_id, user_id),
    ).fetchone()
    if row is None:
        return False

    result = json.loads(row["result_json"])
    for relative_path in result.get("cleanup_paths", []):
        _safe_remove_static_file(relative_path)

    _get_db().execute(
        "DELETE FROM analyses WHERE id = ? AND user_id = ?",
        (analysis_id, user_id),
    )
    _get_db().commit()
    return True


def _clear_history_for_user(user_id: int) -> int:
    rows = _get_db().execute(
        "SELECT id FROM analyses WHERE user_id = ?",
        (user_id,),
    ).fetchall()
    deleted = 0
    for row in rows:
        if _delete_analysis_record(user_id, int(row["id"])):
            deleted += 1
    return deleted


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if g.user is None:
            flash("Please sign in to access the analysis workspace.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


def _render_workspace(
    *,
    selected_model: str = "cardiac",
    result: dict[str, Any] | None = None,
    error_message: str | None = None,
    active_analysis_id: int | None = None,
):
    if selected_model not in MODEL_LOOKUP:
        selected_model = "cardiac"

    history = _fetch_history(int(g.user["id"])) if g.user is not None else []
    return render_template(
        "workspace.html",
        model_cards=MODEL_CARDS,
        selected_card=MODEL_LOOKUP[selected_model],
        selected_model=selected_model,
        result=result,
        error_message=error_message,
        history=history,
        active_analysis_id=active_analysis_id,
        current_user=g.user,
        now_display=_format_timestamp(datetime.now(timezone.utc)),
    )


app = Flask(__name__, static_folder=str(STATIC_DIR), template_folder=str(TEMPLATE_DIR))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.secret_key = os.environ.get("MEDICALAI_SECRET_KEY", "medicalai-final-year-project")

MODELS = ModelRegistry(DEVICE)
init_database()


@app.context_processor
def inject_project_context() -> dict[str, Any]:
    return {
        "project_short_title": PROJECT_SHORT_TITLE,
        "project_subtitle": PROJECT_SUBTITLE,
        "project_full_title": PROJECT_FULL_TITLE,
        "college_name": COLLEGE_NAME,
        "department_name": DEPARTMENT_NAME,
        "guide_name": GUIDE_NAME,
        "coordinator_name": COORDINATOR_NAME,
        "academic_year": ACADEMIC_YEAR,
        "team_members": TEAM_MEMBERS,
    }


@app.before_request
def load_logged_in_user() -> None:
    user_id = session.get("user_id")
    g.user = _fetch_user(int(user_id)) if user_id else None


@app.teardown_appcontext
def close_database(_: BaseException | None) -> None:
    connection = getattr(g, "_database", None)
    if connection is not None:
        connection.close()


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(_: RequestEntityTooLarge):
    if g.user is not None:
        return _render_workspace(error_message="The uploaded file is larger than the 64 MB site limit."), 413
    flash("The uploaded file exceeded the 64 MB site limit.", "error")
    return redirect(url_for("login"))


@app.get("/")
def home():
    return render_template("home.html", model_cards=MODEL_CARDS, current_user=g.user)


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user is not None:
        return redirect(url_for("workspace"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        row = _get_db().execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()

        if row is None or not check_password_hash(row["password_hash"], password):
            flash("Incorrect username or password.", "error")
        else:
            session.clear()
            session["user_id"] = int(row["id"])
            return redirect(url_for("workspace"))

    return render_template("login.html", current_user=g.user)


@app.get("/logout")
def logout():
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("home"))


@app.get("/workspace")
@login_required
def workspace():
    selected_model = request.args.get("model", "cardiac")
    return _render_workspace(selected_model=selected_model)


@app.get("/analysis/<int:analysis_id>")
@login_required
def analysis_detail(analysis_id: int):
    record = _fetch_analysis(int(g.user["id"]), analysis_id)
    if record is None:
        flash("That analysis record could not be found.", "error")
        return redirect(url_for("workspace"))
    result = record["result"]
    result["created_at"] = record["created_at"]
    return _render_workspace(
        selected_model=result["model_key"],
        result=result,
        active_analysis_id=analysis_id,
    )


@app.post("/analysis/<int:analysis_id>/delete")
@login_required
def delete_analysis(analysis_id: int):
    deleted = _delete_analysis_record(int(g.user["id"]), analysis_id)
    if deleted:
        flash("Analysis removed from recent history.", "info")
    else:
        flash("That analysis could not be removed.", "error")
    return redirect(url_for("workspace"))


@app.post("/history/clear")
@login_required
def clear_history():
    deleted = _clear_history_for_user(int(g.user["id"]))
    if deleted:
        flash(f"Cleared {deleted} analysis record(s).", "info")
    else:
        flash("No analysis history was found to clear.", "info")
    return redirect(url_for("workspace"))


@app.post("/run-sample/<model_key>")
@login_required
def run_sample(model_key: str):
    if model_key not in MODEL_LOOKUP:
        return _render_workspace(error_message="Unknown model requested."), 400

    card = MODEL_LOOKUP[model_key]
    if not card.sample_relative:
        return _render_workspace(
            selected_model=model_key,
            error_message="No built-in sample is configured for this model yet.",
        ), 400

    sample_path = STATIC_DIR / card.sample_relative
    if not sample_path.exists():
        return _render_workspace(
            selected_model=model_key,
            error_message="The built-in sample file is not available in this workspace.",
        ), 500

    if model_key == "atrium":
        return _render_workspace(
            selected_model=model_key,
            error_message="Atrium segmentation needs a NIfTI MRI volume and does not currently ship with a built-in sample.",
        ), 400

    image = Image.open(sample_path).convert("RGB")
    uploaded_name = f"{uuid.uuid4().hex}.png"
    image.save(UPLOAD_DIR / uploaded_name, format="PNG")

    result = _run_classifier(model_key, uploaded_name, image)
    analysis_id = _save_analysis(int(g.user["id"]), result)
    return redirect(url_for("analysis_detail", analysis_id=analysis_id))


@app.post("/analyze")
@login_required
def analyze():
    model_key = request.form.get("model", "cardiac")
    if model_key not in MODEL_LOOKUP:
        return _render_workspace(error_message="Please choose one of the available models."), 400

    file_storage = request.files.get("file")
    if file_storage is None or not file_storage.filename:
        return _render_workspace(
            selected_model=model_key,
            error_message="Please choose a file before starting the analysis.",
        ), 400

    if not _allowed_extension(model_key, file_storage.filename):
        return _render_workspace(
            selected_model=model_key,
            error_message=f"{MODEL_LOOKUP[model_key].title} expects {MODEL_LOOKUP[model_key].accepted_formats}.",
        ), 400

    try:
        if model_key == "atrium":
            uploaded_name, upload_path = _save_uploaded_mri(file_storage)
            result = _run_atrium(uploaded_name, upload_path)
        else:
            uploaded_name, image = _save_uploaded_xray(file_storage)
            result = _run_classifier(model_key, uploaded_name, image)

        analysis_id = _save_analysis(int(g.user["id"]), result)
        return redirect(url_for("analysis_detail", analysis_id=analysis_id))
    except ValueError as exc:
        return _render_workspace(selected_model=model_key, error_message=str(exc)), 400
    except RuntimeError as exc:
        return _render_workspace(
            selected_model=model_key,
            error_message=f"Model loading failed: {exc}",
        ), 500
    except Exception as exc:
        return _render_workspace(
            selected_model=model_key,
            error_message=f"Unexpected inference error: {type(exc).__name__}: {exc}",
        ), 500


@app.get("/report/<int:analysis_id>.txt")
@login_required
def report_download(analysis_id: int):
    record = _fetch_analysis(int(g.user["id"]), analysis_id)
    if record is None:
        flash("That report could not be found.", "error")
        return redirect(url_for("workspace"))

    payload = _build_report_text(record).encode("utf-8")
    return send_file(
        io.BytesIO(payload),
        as_attachment=True,
        download_name=f"automated-medical-image-analysis-report-{analysis_id}.txt",
        mimetype="text/plain",
    )


@app.get("/health")
def health():
    MODELS.load()
    return {
        "status": "ready" if MODELS.load_error is None else "error",
        "device": DEVICE.type,
        "load_error": MODELS.load_error,
        "database": str(DATABASE_PATH),
    }


if __name__ == "__main__":
    app.run(debug=True)
@app.route("/")
def home():
    return render_template("index.html")
