from __future__ import annotations

from base64 import b64encode

from flask import Blueprint, current_app, render_template, request
from werkzeug.utils import secure_filename

from .model import classifier

main = Blueprint("main", __name__)


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in current_app.config["ALLOWED_EXTENSIONS"]
    )


@main.route("/", methods=["GET", "POST"])
def index():
    context = {
        "error": None,
        "filename": None,
        "image_data": None,
        "model_name": current_app.config["MODEL_NAME"],
        "predictions": [],
    }

    if request.method == "POST":
        uploaded_file = request.files.get("image")
        if uploaded_file is None or uploaded_file.filename == "":
            context["error"] = "Choose an image before running prediction."
            return render_template("index.html", **context)

        filename = secure_filename(uploaded_file.filename)
        context["filename"] = filename

        if not allowed_file(filename):
            allowed = ", ".join(sorted(current_app.config["ALLOWED_EXTENSIONS"]))
            context["error"] = f"Unsupported file type. Use one of: {allowed}."
            return render_template("index.html", **context)

        raw_bytes = uploaded_file.read()
        if not raw_bytes:
            context["error"] = "The uploaded file is empty."
            return render_template("index.html", **context)

        try:
            context["predictions"] = classifier.predict(
                raw_bytes,
                top=current_app.config["TOP_PREDICTIONS"],
            )
            mime_type = uploaded_file.mimetype or "image/jpeg"
            encoded_image = b64encode(raw_bytes).decode("utf-8")
            context["image_data"] = f"data:{mime_type};base64,{encoded_image}"
        except Exception:
            current_app.logger.exception("Prediction failed for %s", filename)
            context["error"] = (
                "Prediction failed. Upload a valid image and try again."
            )

    return render_template("index.html", **context)
