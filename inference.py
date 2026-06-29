import os
import argparse
import cv2
import torch
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import shapes as rasterio_shapes
from shapely.geometry import shape
from tqdm import tqdm

from models.segformer import SegFormerBinary

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
TILE_SIZE = 512
THRESHOLD = 0.5

os.makedirs('outputs/geojson',  exist_ok=True)
os.makedirs('outputs/overlays', exist_ok=True)

model = SegFormerBinary().to(DEVICE)
model.load_state_dict(torch.load('weights/best_model.pt', map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

def predict(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W  = image.shape[:2]

    image_512 = cv2.resize(image, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_LINEAR)

    image_512 = np.ascontiguousarray(image_512)
    tensor    = torch.tensor(image_512, dtype=torch.float32).permute(2, 0, 1) / 255.0
    tensor    = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits)

    mask_512 = probs.squeeze().cpu().numpy()
    mask     = cv2.resize(mask_512, (W, H), interpolation=cv2.INTER_NEAREST)
    binary   = (mask >= THRESHOLD).astype(np.uint8)

    return binary, image, H, W

def save_geojson(binary_mask, image_path, out_path):
    try:
        with rasterio.open(image_path) as src:
            transform = src.transform
            crs       = src.crs

        polygons = []
        for geom_dict, value in rasterio_shapes(binary_mask, transform=transform):
            if value == 1:
                geom = shape(geom_dict)
                if geom.area > 50:
                    polygons.append(geom)

        gdf = gpd.GeoDataFrame({"geometry": polygons, "class": "turf"}, crs=crs)

        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")

    except Exception:
        polygons = []
        for geom_dict, value in rasterio_shapes(binary_mask):
            if value == 1:
                geom = shape(geom_dict)
                if geom.area > 50:
                    polygons.append(geom)

        gdf = gpd.GeoDataFrame({"geometry": polygons, "class": "turf"})

    gdf.to_file(out_path, driver="GeoJSON")
    print(f"  GeoJSON saved : {out_path}  ({len(gdf)} polygons)")

def save_overlay(image_rgb, binary_mask, out_path):
    overlay = image_rgb.copy()
    green   = np.array([0, 255, 0], dtype=np.uint8)
    overlay[binary_mask == 1] = (
        overlay[binary_mask == 1] * 0.6 + green * 0.4
    ).astype(np.uint8)

    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"  Overlay saved : {out_path}")

def run(image_path):
    name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\nProcessing: {image_path}")

    binary_mask, image_rgb, H, W = predict(image_path)

    save_geojson(binary_mask, image_path, f"outputs/geojson/{name}.geojson")
    save_overlay(image_rgb,   binary_mask, f"outputs/overlays/{name}.png")

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, help="Path to a single image")
parser.add_argument("--input", type=str, help="Path to a folder of images")
args = parser.parse_args()

if args.image:
    run(args.image)

elif args.input:
    images = [
        f for f in os.listdir(args.input)
        if f.endswith((".tif", ".tiff", ".jpg", ".png"))
    ]
    print(f"Found {len(images)} images in {args.input}")
    for fname in tqdm(images, desc="Inference"):
        run(os.path.join(args.input, fname))

else:
    print("Usage:")
    print("  python inference.py --image input_image.tif")
    print("  python inference.py --input ./images/")