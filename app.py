import os
import io
import json
import re
import base64
import numpy as np
import gradio as gr
from PIL import Image, ImageEnhance, ImageFilter
from openai import OpenAI

# Read Hugging Face API token from environment (Space Secret).
hf_token = os.environ.get("HF_TOKEN")

# OpenAI client configured to call Hugging Face router with OpenAI-compatible API.
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=hf_token,
)

# Vision-language model used for reading labels and assessing compliance.
MODEL = "Qwen/Qwen3-VL-8B-Instruct:novita"


def preprocess(img: Image.Image, min_side: int = 900) -> Image.Image:
    """
    Prepare image for the VL model:
    - ensure RGB
    - upscale smaller images
    - sharpen and boost contrast to make text more legible.
    """
    img = img.convert("RGB")
    w, h = img.size

    # Upscale so the shortest side is at least min_side.
    if min(w, h) < min_side:
        scale = min_side / min(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Light sharpening and contrast enhancement.
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageEnhance.Contrast(img).enhance(1.35)
    return img


def img_to_data_uri(img: Image.Image) -> str:
    """
    Encode a PIL image as a JPEG data URI so it can be passed
    via the OpenAI-style 'image_url' interface.
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def make_tiles(img: Image.Image) -> list[tuple[str, Image.Image]]:
    """
    Simple tiling strategy:
    - current version: treat entire image as a single 'full label' tile.
    - kept as a list for future extension (e.g., curved bottle tiling).
    """
    return [("full label", img)]


# System message describing role and response constraints for the model.
SYSTEM = (
    "You are a strict TTB (Alcohol and Tobacco Tax and Trade Bureau) compliance specialist "
    "with expert vision. You read all text from alcohol label images including decorative, "
    "stylized, curved, rotated, and small-print fonts. "
    "Respond ONLY with valid JSON. No markdown fences. No preamble."
)


def build_prompt(tile_names: list[str]) -> str:
    """
    Detailed user prompt:
    - explains what to read from the label
    - defines each TTB element
    - sets confidence/status rules
    - specifies the exact JSON schema to return.
    """
    return """Examine this alcohol label for TTB regulatory compliance.

Read ALL visible text: decorative headers, fine print, back panel, rotated or curved text.

DEFINITIONS:
- brand_name: Fanciful/trade name only (e.g. "Jack Daniel's"). NOT the distillery suffix.
- class_type: TTB class/type (e.g. "Kentucky Straight Bourbon Whiskey").
- alcohol_content: ABV % required. Proof alone is NOT sufficient.
- net_contents: Volume (e.g. 750 mL).
- bottler_address: Full bottler/importer name + street + city + state + ZIP.
- country_of_origin: Imports only. US domestic set status="not_required", value=null.
- government_warning: BOTH statements required:
    (1) "ACCORDING TO THE SURGEON GENERAL, WOMEN SHOULD NOT DRINK ALCOHOLIC BEVERAGES
        DURING PREGNANCY BECAUSE OF THE RISK OF BIRTH DEFECTS."
    (2) "CONSUMPTION OF ALCOHOLIC BEVERAGES IMPAIRS YOUR ABILITY TO DRIVE A CAR OR
        OPERATE MACHINERY, AND MAY CAUSE HEALTH PROBLEMS."
  Use "present" / "unclear" (partial/blurry) / "missing"
  value: clean canonical warning text.

CONFIDENCE: Start 1.0. Subtract 0.1 per "unclear", 0.2 per "missing".

STATUS:
- "compliant"     ALL elements present (country_of_origin may be "not_required")
- "non_compliant" ANY element missing
- "needs_review"  ANY unclear, none missing

Extract all readable text into "ocr_text" (one block per line, top-to-bottom).

Return ONLY this JSON:
{
  "ocr_text": "<all label text, one block per line>",
  "beverage_type": "beer|wine|distilled_spirits|unknown",
  "overall_status": "compliant|non_compliant|needs_review",
  "confidence": 0.85,
  "summary": "<one sentence verdict>",
  "elements": {
    "brand_name":         {"status": "present|missing|unclear", "value": "<text or null>", "notes": ""},
    "class_type":         {"status": "present|missing|unclear", "value": "<text or null>", "notes": ""},
    "alcohol_content":    {"status": "present|missing|unclear", "value": "<text or null>", "notes": ""},
    "net_contents":       {"status": "present|missing|unclear", "value": "<text or null>", "notes": ""},
    "bottler_address":    {"status": "present|missing|unclear", "value": "<text or null>", "notes": ""},
    "country_of_origin":  {"status": "present|missing|unclear|not_required", "value": null, "notes": ""},
    "government_warning": {"status": "present|missing|unclear", "value": "<canonical warning or null>", "notes": ""}
  },
  "issues": ["<issue>"],
  "recommendations": ["<fix>"]
}"""


def analyze_label(img: Image.Image) -> dict:
    """
    Full label analysis pipeline:
    - preprocess image
    - generate tiles (currently just full label)
    - build multimodal content
    - call Qwen via HF router
    - parse and return JSON.
    """
    processed  = preprocess(img)
    tiles      = make_tiles(processed)
    tile_names = [name for name, _ in tiles]

    # Build multimodal content: [panel title, image] pairs + final prompt.
    content = []
    for name, tile in tiles:
        content.append({"type": "text",      "text": f"[{name.upper()}]"})
        content.append({"type": "image_url", "image_url": {"url": img_to_data_uri(tile)}})
    content.append({"type": "text", "text": build_prompt(tile_names)})

    # Call HF router using OpenAI-style Chat Completions API.
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": content},
        ],
        max_tokens=800,
    )

    raw = resp.choices[0].message.content.strip()

    # Strip optional ```json fences if the model includes them.
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$",          "", raw)

    # Try strict JSON parse; if it fails, fall back to first {...} block.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        # Surface a trimmed snippet to aid debugging.
        raise ValueError(f"Non-JSON response:\n{raw[:300]}")


# Icons and labels used in the formatted text report.
STATUS_ICON  = {"compliant": "✅", "non_compliant": "❌", "needs_review": "⚠️"}
ELEMENT_ICON = {"present": "✅", "missing": "❌", "unclear": "⚠️", "not_required": "➖"}
ELEMENT_LABELS = [
    ("brand_name",         "Brand Name"),
    ("class_type",         "Class / Type"),
    ("alcohol_content",    "Alcohol Content"),
    ("net_contents",       "Net Contents"),
    ("bottler_address",    "Bottler Address"),
    ("country_of_origin",  "Country of Origin"),
    ("government_warning", "Govt. Warning"),
]


def format_report(c: dict) -> str:
    """
    Convert the JSON compliance result into a human-readable
    multi-line string for the UI.
    """
    overall = c.get("overall_status", "")
    lines = [
        f"{STATUS_ICON.get(overall, '?')} {overall.replace('_', ' ').upper()}",
        f"Beverage:   {c.get('beverage_type', 'unknown').replace('_', ' ').title()}",
        f"Confidence: {int(c.get('confidence', 0) * 100)}%",
        f"\n{c.get('summary', '')}",
        "\n-- Mandatory Elements --",
    ]

    # Per-element status with icon, extracted value, and notes.
    for key, label in ELEMENT_LABELS:
        el   = c.get("elements", {}).get(key, {})
        st   = el.get("status", "")
        icon = ELEMENT_ICON.get(st, "?")
        val  = f': "{el["value"]}"' if el.get("value") else ""
        note = f' -- {el["notes"]}' if el.get("notes") and st != "present" else ""
        lines.append(f"  {icon} {label}{val}{note}")

    # Optional issues and recommendations sections.
    if c.get("issues"):
        lines.append("\n-- Issues --")
        for i in c["issues"]:
            lines.append(f"  ! {i}")

    if c.get("recommendations"):
        lines.append("\n-- Recommendations --")
        for r in c["recommendations"]:
            lines.append(f"  > {r}")

    return "\n".join(lines)


def analyze(image: np.ndarray):
    """
    Gradio callback:
    - validates image and token
    - runs analysis
    - returns status, OCR text, and formatted report.
    """
    if image is None:
        return "Please upload a label image.", "", ""

    if not hf_token:
        return "HF_TOKEN not set -- add it in Space Settings > Secrets.", "", ""

    try:
        # Convert NumPy array from Gradio into PIL Image for processing.
        result = analyze_label(Image.fromarray(image))
    except Exception as e:
        return f"Analysis failed: {e}", "", ""

    # Extract OCR text from the model output and remove it from the dict.
    ocr_text = result.pop("ocr_text", "(not extracted)")

    # Map internal status codes to human-friendly labels.
    status   = {
        "compliant":     "Compliant",
        "non_compliant": "Non-Compliant",
        "needs_review":  "Needs Review",
    }.get(result.get("overall_status", ""), "Unknown")

    return status, ocr_text, format_report(result)


# Custom CSS to style the Gradio UI like a classic spirits label.
LABEL_CSS = """
<style>
  *, *::before, *::after {
    font-family: 'Times New Roman', Times, serif !important;
    box-sizing: border-box;
  }
  body, .gradio-container, .gradio-container > .main, footer {
    background-color: #F5F0E0 !important;
  }
  .label-header {
    background-color: #1A1008;
    border: 2px solid #C9A84C;
    padding: 20px 28px 16px;
    margin-bottom: 20px;
    text-align: center;
  }
  .label-header h1 {
    color: #C9A84C !important;
    font-size: 1.75rem !important;
    letter-spacing: 0.2em !important;
    margin: 0 0 5px !important;
    font-weight: bold !important;
  }
  .label-header p {
    color: #A8893A !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.28em !important;
    margin: 0 !important;
  }
  .gradio-container [data-testid="image"],
  .gradio-container .upload-container {
    border: 1.5px dashed #A8893A !important;
    background-color: #FDFAF2 !important;
    border-radius: 2px !important;
  }
  .gradio-container .upload-container span,
  .gradio-container [data-testid="image"] span,
  .gradio-container .drop-text,
  .gradio-container .upload-container p,
  .gradio-container [data-testid="image"] p {
    color: #2C1A0E !important;
    opacity: 1 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.06em !important;
  }
  .gradio-container .upload-container svg,
  .gradio-container [data-testid="image"] svg {
    color: #A8893A !important;
    opacity: 1 !important;
  }
  .gradio-container textarea,
  .gradio-container input[type="text"] {
    background-color: #FDFAF2 !important;
    color: #2C1A0E !important;
    border: 1.5px solid #A8893A !important;
    border-radius: 2px !important;
    font-family: 'Times New Roman', Times, serif !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
    overflow: hidden !important;
    resize: none !important;
  }
  .gradio-container textarea[aria-label="Compliance Status"] {
    background-color: #1A1008 !important;
    color: #C9A84C !important;
    border: 1.5px solid #C9A84C !important;
    font-weight: bold !important;
    letter-spacing: 0.08em !important;
    font-size: 1rem !important;
  }
  .gradio-container .label-wrap span,
  .gradio-container .block label span {
    color: #1A1008 !important;
    font-weight: bold !important;
    letter-spacing: 0.1em !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
  }
  .gradio-container button.primary,
  .gradio-container button[variant="primary"] {
    background-color: #1A1008 !important;
    color: #C9A84C !important;
    border: 1.5px solid #C9A84C !important;
    border-radius: 2px !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.14em !important;
    padding: 12px 22px !important;
    width: 100% !important;
    transition: background 0.2s, color 0.2s;
  }
  .gradio-container button.primary:hover {
    background-color: #C9A84C !important;
    color: #1A1008 !important;
  }
  .gradio-container .block,
  .gradio-container .gr-box,
  .gradio-container .form {
    background-color: #F5F0E0 !important;
    border-color: #A8893A !important;
  }
  .footer-note {
    border-top: 1px solid #A8893A;
    margin-top: 20px;
    padding-top: 12px;
    color: #2C1A0E !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    text-align: center;
    line-height: 1.9;
  }
  .footer-note::before {
    content: '';
    display: block;
    width: 40px;
    border-top: 2px solid #C9A84C;
    margin: 0 auto 10px;
  }
</style>
"""


# Build the Gradio UI using a custom theme + CSS for a label-like look.
with gr.Blocks(title="TTB Label Analyzer", theme=gr.themes.Base()) as demo:

    # Inject custom styles.
    gr.HTML(LABEL_CSS)

    # Top header styled like a spirits label band.
    gr.HTML("""
    <div class="label-header">
      <h1>TTB LABEL ANALYZER</h1>
      <p>ALCOHOL LABEL COMPLIANCE &nbsp;&middot;&nbsp; TTB REGULATORY REVIEW</p>
    </div>
    """)

    with gr.Row():
        # Left column: image upload and analyze button.
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Upload Label Image", height=380)
            analyze_btn = gr.Button("Analyze Label", variant="primary")

        # Right column: status, extracted OCR text, and full report.
        with gr.Column():
            status_out = gr.Textbox(
                label="Compliance Status",
                lines=1, max_lines=2,
                interactive=False
            )
            ocr_out = gr.Textbox(
                label="Extracted Text (OCR)",
                lines=4, max_lines=30,
                interactive=False
            )
            report_out = gr.Textbox(
                label="Full Compliance Report",
                lines=6, max_lines=60,
                interactive=False
            )

    # Wire the button click to the analysis function.
    analyze_btn.click(fn=analyze, inputs=image_input, outputs=[status_out, ocr_out, report_out])

    # Footer reminding users of core TTB-required elements.
    gr.HTML("""
    <div class="footer-note">
      TTB REQUIRED ELEMENTS &nbsp;&middot;&nbsp; BRAND NAME &nbsp;&middot;&nbsp; CLASS / TYPE
      &nbsp;&middot;&nbsp; ALCOHOL CONTENT &nbsp;&middot;&nbsp; NET CONTENTS
      &nbsp;&middot;&nbsp; BOTTLER ADDRESS &nbsp;&middot;&nbsp; COUNTRY OF ORIGIN
      &nbsp;&middot;&nbsp; GOVERNMENT WARNING
    </div>
    """)

# Launch the app (Spaces or local).
demo.launch(server_name="0.0.0.0", server_port=7860)