import base64
import datetime as dt
import io
import json
import os
import uuid
from pathlib import Path
from urllib.parse import urlencode

import requests
from google import genai
from google.genai import types
from flask import Flask, jsonify, request, send_file
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pymongo import ASCENDING, MongoClient
from pymongo.errors import ServerSelectionTimeoutError






try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


APP_NAME = "LandscapeVision.AI - Demo"
OUTPUT_DIR = Path(os.getenv("LANDSCAPE_OUTPUT_DIR", "generated_designs"))
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "landscape_vision_demo")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
LANDSCAPE_SIZE = (1024, 576)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
MONGO_READY = False


THEMES = [
    {
        "id": "modern-oasis",
        "name": "Modern Oasis",
        "description": "Clean lines, statement palms, low maintenance",
        "plantIds": ["date-palm", "washingtonia", "bougainvillea", "bermuda"],
    },
    {
        "id": "heritage-courtyard",
        "name": "Heritage Courtyard",
        "description": "Olive, sidr & ghaf — timeless and shaded",
        "plantIds": ["olive-tree", "sidr", "ghaf", "oleander"],
    },
    {
        "id": "flowering-retreat",
        "name": "Flowering Retreat",
        "description": "Color year-round with hardy blooms",
        "plantIds": ["bougainvillea", "desert-rose", "oleander", "neem"],
    },
    {
        "id": "highland-garden",
        "name": "Highland Garden",
        "description": "Cool-climate picks for Abha & Taif",
        "plantIds": ["jacaranda", "lavender", "olive-tree", "bermuda"],
    },
]


SEED_INVENTORY = [
    ("olive-tree", "Olive Tree", "tree", "mediterranean,modern-minimal", "hot-dry,mediterranean", 35, 95),
    ("lavender", "Lavender", "shrub", "mediterranean,modern-minimal", "hot-dry,mediterranean,temperate", 140, 8),
    ("rosemary", "Rosemary", "shrub", "mediterranean,desert-oasis", "hot-dry,mediterranean", 90, 7),
    ("agave", "Agave", "succulent", "desert-oasis,modern-minimal", "hot-dry", 65, 18),
    ("date-palm", "Date Palm", "tree", "desert-oasis,mediterranean", "hot-dry", 12, 180),
    ("ficus-nitida", "Ficus Nitida", "tree", "lush-family,modern-minimal", "hot-dry,subtropical,temperate", 44, 70),
    ("bougainvillea", "Bougainvillea", "climber", "mediterranean,lush-family", "hot-dry,mediterranean,subtropical", 75, 12),
    ("jasmine", "Star Jasmine", "climber", "lush-family,mediterranean", "mediterranean,temperate,subtropical", 80, 10),
    ("dwarf-grass", "Dwarf Lawn Grass", "groundcover", "lush-family", "temperate,subtropical", 250, 5),
    ("boxwood", "Boxwood", "shrub", "modern-minimal,lush-family", "temperate,mediterranean", 120, 11),
    ("porcelain-tile", "Porcelain Outdoor Tile", "tile", "modern-minimal,lush-family", "hot-dry,mediterranean,temperate,subtropical", 500, 19),
    ("natural-stone", "Natural Stone Paver", "paver", "mediterranean,desert-oasis,lush-family", "hot-dry,mediterranean,temperate", 420, 22),
    ("pergola", "Aluminum Pergola", "shade", "mediterranean,modern-minimal,lush-family", "hot-dry,mediterranean,temperate,subtropical", 8, 650),
]


def get_db():
    return mongo_client[MONGO_DB_NAME]

def get_collection(name):
    return get_db()[name]

def init_db():
    global MONGO_READY
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        mongo_client.admin.command("ping")
        db = get_db()
        db.inventory.create_index([("id", ASCENDING)], unique=True)
        db.projects.create_index([("id", ASCENDING)], unique=True)
        db.designs.create_index([("id", ASCENDING)], unique=True)

        for row in SEED_INVENTORY:
            item = {
                "id": row[0],
                "name": row[1],
                "category": row[2],
                "themes": [t.strip() for t in row[3].split(",")],
                "climates": [c.strip() for c in row[4].split(",")],
                "stock": row[5],
                "unit_price": row[6],
            }
            db.inventory.update_one({"id": item["id"]}, {"$set": item}, upsert=True)

        MONGO_READY = True
        print("MongoDB connected and initialized")
    except ServerSelectionTimeoutError as exc:
        print(f"MongoDB error: {str(exc)}")


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def json_response(payload, status=200):
    response = jsonify(payload)
    response.status_code = status
    return response

def placeholder_design(prompt, theme_id):
    img = Image.new("RGB", LANDSCAPE_SIZE, color=(100, 150, 100))
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), f"Design for {theme_id}", fill=(255, 255, 255))
    draw.text((50, 100), f"Theme: {theme_id}", fill=(255, 255, 255))
    return img

def fit_landscape_size(image):
    if image.size != LANDSCAPE_SIZE:
        image = image.resize(LANDSCAPE_SIZE, Image.Resampling.LANCZOS)
    return image

def save_image(image, prefix):
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    filepath = OUTPUT_DIR / filename
    image.save(filepath)
    return str(filepath)

def call_gemini_image(prompt, landscape_image_base64=None):
    if not GEMINI_API_KEY:
        return placeholder_design(prompt, "demo")

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        contents = []

        if landscape_image_base64:
            if "," in landscape_image_base64:
                landscape_image_base64 = landscape_image_base64.split(",", 1)[1]
            image_bytes = base64.b64decode(landscape_image_base64)
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            contents.append(pil_image)

        contents.append(prompt)

        print(f"--- Gemini request: model={GEMINI_IMAGE_MODEL}, parts={len(contents)} ---")

        response = client.models.generate_content(
            model=GEMINI_IMAGE_MODEL,
            contents=contents
        )

        print(f"--- Gemini raw response ---")
        print(f"Candidates count: {len(response.candidates)}")
        for i, candidate in enumerate(response.candidates):
            print(f"Candidate {i}: finish_reason={candidate.finish_reason}")
            for j, part in enumerate(candidate.content.parts):
                has_image = bool(part.inline_data and part.inline_data.data)
                text_preview = part.text[:100] if part.text else None
                print(f"  Part {j}: inline_data={has_image}, text={text_preview}")
        print(f"--- End Gemini response ---")

        for part in response.candidates[0].content.parts:
            if part.inline_data:
                image = Image.open(io.BytesIO(part.inline_data.data))
                return fit_landscape_size(image)

        print("No image part in Gemini response")
        return placeholder_design(prompt, "no-image")

    except Exception as e:
        print(f"Gemini exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return placeholder_design(prompt, "failed")


def build_prompt(theme, selected_items, preferences, location=None):
    theme_name = theme.get("name", theme["id"])
    theme_desc = theme.get("description", "")
    plant_names = [item.get("name", item["id"]) for item in selected_items[:8]]
    plant_list = "\n".join(f"- {name}" for name in plant_names) if plant_names else "- Natural plants suited to this theme"

    return f"""You are an expert landscape architect and photorealistic image editor with 20 years of experience.

You have been given a photograph of an outdoor landscape. Your task is to enhance this exact photo by adding plants and greenery into the scene.

STRICT RULES — YOU MUST FOLLOW THESE:
1. DO NOT change the background, sky, ground, walls, paths, or any existing structures
2. DO NOT alter the lighting, shadows, perspective, or camera angle
3. DO NOT move or remove anything already in the image
4. ONLY add the specified plants — place them naturally within the existing scene
5. The final image must look like a real photograph, not a render or illustration

DESIGN THEME: {theme_name}
{f"THEME STYLE: {theme_desc}" if theme_desc else ""}

PLANTS TO ADD TO THIS LANDSCAPE:
{plant_list}

PLACEMENT INSTRUCTIONS:
- Study the landscape structure carefully — identify ground areas, edges, walls, and open spaces
- Place each plant in a realistic position where it would naturally grow (ground level, near walls, along paths, in open soil areas)
- Respect perspective and depth — plants closer to camera appear larger, plants further away appear smaller
- Match the lighting and shadows of the existing photo when placing plants
- Ensure plants look rooted in the ground, not floating or pasted on top
- Use the correct scale for each plant type (trees are tall, shrubs are medium, groundcover is low)
{f"CLIENT PREFERENCES: {preferences}" if preferences else ""}

OUTPUT: Return the same landscape photograph with the plants added naturally. The image must be photorealistic and indistinguishable from a real garden photo."""


@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        return ("", 204)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = os.getenv("CORS_ORIGIN", "*")
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    return response

@app.route("/", methods=["GET"])
def index():
    return json_response({
        "name": APP_NAME,
        "status": "ready for demo",
        "no_auth": True,
        "no_payment": True,
        "quick_flow": "POST /projects → POST /projects/<id>/generate → GET /designs/<id>",
        "routes": {
            "health": "GET /health",
            "themes": "GET /themes",
            "inventory": "GET /inventory?theme_id=X&climate_zone=Y",
            "create_project": "POST /projects (no auth needed)",
            "generate_design": "POST /projects/<id>/generate",
            "get_design": "GET /designs/<id>",
            "get_image": "GET /designs/<id>/image",
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return json_response({
        "ok": MONGO_READY,
        "app": APP_NAME,
        "database": "mongodb",
        "gemini_available": bool(GEMINI_API_KEY)
    }, 200 if MONGO_READY else 503)

@app.route("/themes", methods=["GET"])
def themes():
    return json_response({"themes": THEMES})

@app.route("/inventory", methods=["GET"])
def inventory():
    theme_id = request.args.get("theme_id")
    climate_zone = request.args.get("climate_zone")

    query = {}
    if theme_id:
        query["themes"] = theme_id

    items = get_collection("inventory").find(query)
    result = []

    for item in items:
        if climate_zone and climate_zone not in item.get("climates", []):
            item["climate_mismatch"] = True
        else:
            item["climate_mismatch"] = False

        item.pop("_id", None)
        result.append(item)

    return json_response({"items": result})

@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return json_response({"error": "No file provided"}, 400)

    file = request.files["file"]
    if file.filename == "":
        return json_response({"error": "No file selected"}, 400)

    allowed_extensions = {"jpg", "jpeg", "png", "gif", "webp"}
    if not ("." in file.filename and file.filename.rsplit(".", 1)[1].lower() in allowed_extensions):
        return json_response({"error": "Only image files allowed (jpg, jpeg, png, gif, webp)"}, 400)

    try:
        file_bytes = file.read()

        mime_type = "image/jpeg"
        if file.filename.lower().endswith(".png"):
            mime_type = "image/png"
        elif file.filename.lower().endswith(".gif"):
            mime_type = "image/gif"
        elif file.filename.lower().endswith(".webp"):
            mime_type = "image/webp"

        base64_string = base64.b64encode(file_bytes).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{base64_string}"

        return json_response({
            "success": True,
            "landscape_image": data_uri,
            "filename": file.filename,
            "size_bytes": len(file_bytes),
            "mime_type": mime_type,
            "instructions": "Copy the 'landscape_image' value and use it in the /generate endpoint body"
        })

    except Exception as e:
        return json_response({"error": f"Upload failed: {str(e)}"}, 500)

@app.route("/projects", methods=["POST"])
def create_project():
    data = request.get_json(silent=True) or {}
    theme_id = data.get("theme_id")

    if not any(t["id"] == theme_id for t in THEMES):
        return json_response({"error": "Valid theme_id required"}, 400)

    project_id = str(uuid.uuid4())
    theme = next((t for t in THEMES if t["id"] == theme_id), {})

    project = {
        "id": project_id,
        "theme_id": theme_id,
        "climate_zone": data.get("climate_zone", "temperate"),
        "location": data.get("location", {}),
        "preferences": data.get("preferences", ""),
        "selected_item_ids": data.get("selected_item_ids", []),
        "uploaded_images_base64": data.get("uploaded_images_base64", []),
        "created_at": now_iso(),
    }

    get_collection("projects").insert_one(project)

    return json_response({
        "project_id": project_id,
        "theme": theme,
        "status": "ready to generate",
        "next": f"POST /projects/{project_id}/generate"
    }, 201)

@app.route("/projects/<project_id>/generate", methods=["POST"])
def generate_design(project_id):
    project = get_collection("projects").find_one({"id": project_id})
    if not project:
        return json_response({"error": "Project not found"}, 404)

    theme = next((t for t in THEMES if t["id"] == project["theme_id"]), {})
    selected_ids = project.get("selected_item_ids", [])
    selected_items = list(get_collection("inventory").find({"id": {"$in": selected_ids}})) if selected_ids else []

    prompt = build_prompt(theme, selected_items, project.get("preferences", ""), project.get("location"))

    landscape_image = None
    uploaded_images = project.get("uploaded_images_base64", [])
    if uploaded_images and len(uploaded_images) > 0:
        landscape_image = uploaded_images[0]

    design_image = call_gemini_image(prompt, landscape_image)
    image_path = save_image(design_image, "design")
    clean_image_path = save_image(design_image, "clean_design")

    design_id = str(uuid.uuid4())
    design = {
        "id": design_id,
        "project_id": project_id,
        "image_path": image_path,
        "clean_image_path": clean_image_path,
        "paid": False,
        "created_at": now_iso(),
    }

    get_collection("designs").insert_one(design)

    with open(image_path, "rb") as fh:
        image_base64 = base64.b64encode(fh.read()).decode()

    return json_response({
        "design": {
            "id": design_id,
            "project_id": project_id,
            "width": LANDSCAPE_SIZE[0],
            "height": LANDSCAPE_SIZE[1],
            "created_at": design["created_at"],
            "image_url": f"/designs/{design_id}/image",
            "image_base64": image_base64,
        }
    }, 201)

@app.route("/generate", methods=["POST"])
def quick_generate():
    data = request.get_json(silent=True) or {}

    theme_id = data.get("theme_id")
    if not any(t["id"] == theme_id for t in THEMES):
        return json_response({"error": "Valid theme_id required"}, 400)

    landscape_image_base64 = data.get("landscape_image", "")
    if not landscape_image_base64:
        return json_response({"error": "landscape_image (base64) required"}, 400)

    preferences = data.get("preferences", "Modern landscape design")
    location = data.get("location", {})

    theme = next((t for t in THEMES if t["id"] == theme_id), {})
    default_plant_ids = theme.get("plantIds", [])

    selected_item_ids = data.get("selected_item_ids") or default_plant_ids

    selected_items = []
    if selected_item_ids:
        selected_items = list(get_collection("inventory").find({"id": {"$in": selected_item_ids}}))

    prompt = build_prompt(theme, selected_items, preferences, location)

    design_image = call_gemini_image(prompt, landscape_image_base64)
    image_path = save_image(design_image, "design")

    design_id = str(uuid.uuid4())
    design = {
        "id": design_id,
        "theme_id": theme_id,
        "image_path": str(image_path),
        "created_at": now_iso(),
    }
    get_collection("designs").insert_one(design)

    with open(image_path, "rb") as fh:
        image_base64 = base64.b64encode(fh.read()).decode()

    return json_response({
        "success": True,
        "design": {
            "id": design_id,
            "theme": theme.get("name", theme_id),
            "width": LANDSCAPE_SIZE[0],
            "height": LANDSCAPE_SIZE[1],
            "image_base64": image_base64,
            "image_url": f"/designs/{design_id}/image",
        }
    }, 201)

@app.route("/designs/<design_id>", methods=["GET"])
def get_design(design_id):
    design = get_collection("designs").find_one({"id": design_id})
    if not design:
        return json_response({"error": "Design not found"}, 404)

    return json_response({
        "design": {
            "id": design["id"],
            "project_id": design.get("project_id"),
            "width": LANDSCAPE_SIZE[0],
            "height": LANDSCAPE_SIZE[1],
            "created_at": design["created_at"],
            "image_url": f"/designs/{design_id}/image",
        }
    })

@app.route("/designs/<design_id>/image", methods=["GET"])
def get_design_image(design_id):
    design = get_collection("designs").find_one({"id": design_id})
    if not design:
        return json_response({"error": "Design not found"}, 404)

    image_path = design.get("clean_image_path") or design.get("image_path")
    if not image_path or not Path(image_path).exists():
        return json_response({"error": "Image file not found"}, 404)

    return send_file(image_path, mimetype="image/png")


if __name__ == "__main__":
    init_db()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    app.run(host=host, port=port, debug=debug)
