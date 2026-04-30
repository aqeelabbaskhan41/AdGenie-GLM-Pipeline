"""
GLM-Image Server — Fixed I2I pipeline
"""

import io
import gc
import os
import base64
import numpy as np
from typing import Optional, List
from contextlib import asynccontextmanager

import torch
from PIL import Image
from pydantic import BaseModel, field_validator

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File, Form
import uvicorn

MODEL_PATH = os.getenv("MODEL_PATH", "zai-org/GLM-Image")
COMPILE    = os.getenv("TORCH_COMPILE", "0") == "1"


# ══════════════════════════════════════════════════════════════════════════════
#  Request schemas
# ══════════════════════════════════════════════════════════════════════════════

class T2IRequest(BaseModel):
    prompt:              str
    height:              int   = 40 * 32
    width:               int   = 30 * 32
    num_inference_steps: int   = 30
    guidance_scale:      float = 1.5
    seed:                Optional[int] = None

    @field_validator("height", "width")
    @classmethod
    def must_be_mult32(cls, v: int) -> int:
        if v % 32 != 0:
            raise ValueError(f"Dimension {v} must be divisible by 32")
        return v


class I2IRequest(BaseModel):
    prompt:              str
    image_base64:        str
    height:              int   = 40 * 32
    width:               int   = 30 * 32
    num_inference_steps: int   = 30
    guidance_scale:      float = 1.5
    seed:                Optional[int] = None
    strength:            float = 0.8  # How much to transform (0.0 = no change, 1.0 = full change)

    @field_validator("height", "width")
    @classmethod
    def must_be_mult32(cls, v: int) -> int:
        if v % 32 != 0:
            raise ValueError(f"Dimension {v} must be divisible by 32")
        return v


class AdRequest(BaseModel):
    scene:           str
    style:           str  = "professional poster, clean modern layout, vibrant colors, high fidelity"
    aspect:          str  = "portrait"
    headline:        str
    subheadline:     Optional[str] = None
    body_lines:      List[str]     = []
    cta:             Optional[str] = None
    footer:          Optional[str] = None
    color_palette:   Optional[str] = None
    layout_hint:     Optional[str] = None
    num_inference_steps: int   = 30
    guidance_scale:      float = 1.5
    seed:                Optional[int] = None


# ══════════════════════════════════════════════════════════════════════════════
#  Advertisement prompt builder
# ══════════════════════════════════════════════════════════════════════════════

ASPECT_DIMS = {
    "portrait":  (30, 40),
    "landscape": (40, 30),
    "square":    (32, 32),
    "wide":      (48, 27),
    "a4":        (27, 38),
}


def _q(text: str) -> str:
    return f'"{text.strip().strip(chr(34)).strip(chr(39))}"'


def build_ad_prompt(req: AdRequest) -> tuple[str, int, int]:
    w_mult, h_mult = ASPECT_DIMS.get(req.aspect, (30, 40))
    width  = w_mult * 32
    height = h_mult * 32

    parts = [f"bold headline {_q(req.headline)} placed prominently"]
    if req.subheadline:
        parts.append(f"subheadline {_q(req.subheadline)} below the headline")
    for line in req.body_lines:
        parts.append(f"body text line {_q(line)}")
    if req.cta:
        parts.append(f"call-to-action text {_q(req.cta)} on a distinct button or banner")
    if req.footer:
        parts.append(f"footer text {_q(req.footer)}")

    text_block   = "; ".join(parts)
    color_str    = f" Color palette: {req.color_palette}." if req.color_palette else ""
    layout_str   = (f" Layout: {req.layout_hint}."
                    if req.layout_hint
                    else " Centered hierarchy, generous whitespace, clear visual flow.")

    prompt = (
        f"A {req.style} advertisement poster. "
        f"Scene: {req.scene}. "
        f"The poster contains the following text rendered accurately and legibly: {text_block}. "
        f"Typography is clean, modern, and highly readable at all sizes. "
        f"No spelling errors, no blurry text, no garbled characters."
        f"{color_str}{layout_str}"
    )

    return prompt, width, height


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _b64_to_pil(b64: str) -> Image.Image:
    """Convert base64 to PIL Image with proper format handling"""
    img_data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(img_data))
    
    # Convert to RGB if necessary (remove alpha channel)
    if img.mode in ('RGBA', 'LA', 'P'):
        # Create white background for transparency
        if img.mode == 'P':
            img = img.convert('RGBA')
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'RGBA':
            background.paste(img, mask=img.split()[-1])
        else:
            background.paste(img)
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    return img


def _resize_for_i2i(img: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """
    Resize image for I2I pipeline.
    GLM-Image expects the input image to match the target dimensions.
    """
    # Resize to target dimensions
    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return img


def _log_vram(tag: str = ""):
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  [{tag}] VRAM: {used:.1f} / {total:.1f} GB")


def _load_pipe():
    from diffusers.pipelines.glm_image import GlmImagePipeline

    n_gpus     = torch.cuda.device_count()
    total_vram = (
        sum(torch.cuda.get_device_properties(i).total_memory for i in range(n_gpus))
        // (1024 ** 3)
    ) if n_gpus > 0 else 0

    print(f"  GPUs: {n_gpus}   Total VRAM: {total_vram} GB")

    if n_gpus == 0:
        raise RuntimeError("No CUDA GPU detected. This server requires a GPU.")

    if n_gpus >= 2 and total_vram >= 80:
        strategy = "device_map='balanced' (multi-GPU, no offload)"
        pipe = GlmImagePipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
        )
    else:
        strategy = "device_map='cuda' (single GPU, full on-device)"
        pipe = GlmImagePipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

    print(f"  Strategy: {strategy}")

    if COMPILE:
        print("  torch.compile: ENABLED (first inference will be slow)")
        pipe.transformer = torch.compile(
            pipe.transformer,
            mode="reduce-overhead",
            fullgraph=False,
        )

    _log_vram("after load")
    return pipe


def _encode(img: Image.Image, prompt_used: str = "") -> dict:
    payload = {
        "success":      True,
        "image_base64": _pil_to_b64(img),
        "format":       "PNG",
        "width":        img.size[0],
        "height":       img.size[1],
    }
    if prompt_used:
        payload["prompt_used"] = prompt_used
    return payload


# ══════════════════════════════════════════════════════════════════════════════
#  Global pipe instance
# ══════════════════════════════════════════════════════════════════════════════

_pipe_instance = None

def get_pipe():
    global _pipe_instance
    if _pipe_instance is None:
        print("\n" + "="*65)
        print("  Loading GLM-Image pipeline (this may take 30-60 seconds)...")
        print("="*65)
        _pipe_instance = _load_pipe()
        print("\n  ✓ Pipeline ready!\n" + "="*65 + "\n")
    return _pipe_instance


# ══════════════════════════════════════════════════════════════════════════════
#  Inference functions
# ══════════════════════════════════════════════════════════════════════════════

def make_gen(seed: Optional[int]):
    return torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None


def t2i_inference(req: T2IRequest) -> Image.Image:
    pipe = get_pipe()
    print(f"[T2I] {req.width}×{req.height}  steps={req.num_inference_steps}")
    print(f"  {req.prompt[:140]}")
    try:
        result = pipe(
            prompt=req.prompt,
            height=req.height,
            width=req.width,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            generator=make_gen(req.seed),
        )
        img = result.images[0]
        print(f"[T2I] Done ✓  {img.size[0]}×{img.size[1]} px")
        _log_vram("T2I")
        return img
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        raise RuntimeError("GPU OOM — reduce height/width or num_inference_steps")


def i2i_inference(req: I2IRequest) -> Image.Image:
    pipe = get_pipe()
    print(f"[I2I] {req.width}×{req.height}  steps={req.num_inference_steps}, strength={req.strength}")
    
    # Load and prepare input image
    input_img = _b64_to_pil(req.image_base64)
    print(f"  Input image size: {input_img.size[0]}×{input_img.size[1]}")
    
    # Resize to match target dimensions (required by GLM-Image)
    input_img = _resize_for_i2i(input_img, req.width, req.height)
    print(f"  Resized to: {input_img.size[0]}×{input_img.size[1]}")
    
    try:
        # GLM-Image I2I expects the image parameter to be a list
        result = pipe(
            prompt=req.prompt,
            image=[input_img],  # Must be a list
            height=req.height,
            width=req.width,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            strength=req.strength,  # Control transformation strength
            generator=make_gen(req.seed),
        )
        img = result.images[0]
        print(f"[I2I] Done ✓  {img.size[0]}×{img.size[1]} px")
        _log_vram("I2I")
        return img
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        raise RuntimeError("GPU OOM — reduce resolution or num_inference_steps")
    except Exception as e:
        print(f"[I2I] Error: {e}")
        # Fallback: try without strength parameter
        try:
            print("  Retrying without strength parameter...")
            result = pipe(
                prompt=req.prompt,
                image=[input_img],
                height=req.height,
                width=req.width,
                num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
                generator=make_gen(req.seed),
            )
            img = result.images[0]
            print(f"[I2I] Done ✓ (fallback) {img.size[0]}×{img.size[1]} px")
            return img
        except Exception as e2:
            raise RuntimeError(f"I2I failed: {str(e2)}")


def ad_inference(req: AdRequest) -> tuple[Image.Image, str]:
    pipe = get_pipe()
    prompt, width, height = build_ad_prompt(req)
    print(f"[AD]  {width}×{height}  steps={req.num_inference_steps}")
    print(f"  {prompt[:200]}")
    try:
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            generator=make_gen(req.seed),
        )
        img = result.images[0]
        print(f"[AD]  Done ✓  {img.size[0]}×{img.size[1]} px")
        _log_vram("AD")
        return img, prompt
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        raise RuntimeError("GPU OOM — reduce steps or aspect ratio size")


# ══════════════════════════════════════════════════════════════════════════════
#  FastAPI app
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup"""
    print("\n" + "="*65)
    print("  Starting GLM-Image Server")
    print("="*65)
    get_pipe()
    yield
    global _pipe_instance
    _pipe_instance = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan, title="GLM-Image Server", version="1.0.0")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict/ad")
async def predict_ad(req: AdRequest):
    try:
        img, prompt_used = ad_inference(req)
        torch.cuda.empty_cache()
        return _encode(img, prompt_used)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(exc)},
        )


@app.post("/predict/i2i")
async def predict_i2i(req: I2IRequest):
    try:
        img = i2i_inference(req)
        torch.cuda.empty_cache()
        return _encode(img)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(exc)},
        )


@app.post("/predict")
async def predict_t2i(req: T2IRequest):
    try:
        img = t2i_inference(req)
        torch.cuda.empty_cache()
        return _encode(img)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(exc)},
        )


# Alternative I2I endpoint with file upload (easier for testing)
@app.post("/predict/i2i/upload")
async def predict_i2i_upload(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    height: int = Form(1280),
    width: int = Form(960),
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(1.5),
    strength: float = Form(0.8),
    seed: Optional[int] = Form(None),
):
    try:
        # Read and convert uploaded image
        img_data = await image.read()
        img_b64 = base64.b64encode(img_data).decode()
        
        req = I2IRequest(
            prompt=prompt,
            image_base64=img_b64,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed,
        )
        
        img = i2i_inference(req)
        torch.cuda.empty_cache()
        return _encode(img)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(exc)},
        )


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": _pipe_instance is not None}


@app.get("/info")
async def info():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    vram_total = (torch.cuda.get_device_properties(0).total_memory // (1024**3)
                  if torch.cuda.is_available() else 0)
    vram_used = (torch.cuda.memory_allocated() // (1024**3)
                 if torch.cuda.is_available() else 0)
    return {
        "model": MODEL_PATH,
        "gpu": gpu_name,
        "vram_total": f"{vram_total} GB",
        "vram_used": f"{vram_used} GB",
        "compile": COMPILE,
        "endpoints": {
            "POST /predict": "Text-to-Image",
            "POST /predict/i2i": "Image-to-Image",
            "POST /predict/i2i/upload": "Image-to-Image (file upload)",
            "POST /predict/ad": "Advertisement builder",
        },
        "ad_aspects": list(ASPECT_DIMS.keys()),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    args = parser.parse_args()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"
    print("=" * 65)
    print(f"  GLM-Image Server")
    print(f"  Model : {MODEL_PATH}")
    print(f"  GPU   : {gpu_name}")
    print(f"  Host  : {args.host}")
    print(f"  Port  : {args.port}")
    print(f"  Compile: {'ON' if COMPILE else 'OFF'}")
    print("=" * 65)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
    )