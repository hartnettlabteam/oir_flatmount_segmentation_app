import os
import glob
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import cv2
import json

from model import AttentionUNet
from model_transformer import AttentionUNetTransformer
from dataset import _read_image_gray_any, _resize_gray
from vis import overlay_mask, blend_three


def resolve_device(requested: str = "auto") -> torch.device:
	"""Resolve runtime device with CUDA > MPS > CPU fallback."""
	req = (requested or "auto").lower()
	has_cuda = torch.cuda.is_available()
	has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

	if req == "cuda":
		return torch.device("cuda" if has_cuda else ("mps" if has_mps else "cpu"))
	if req == "mps":
		return torch.device("mps" if has_mps else ("cuda" if has_cuda else "cpu"))
	if req == "cpu":
		return torch.device("cpu")
	# auto
	if has_cuda:
		return torch.device("cuda")
	if has_mps:
		return torch.device("mps")
	return torch.device("cpu")


def binarize(x: np.ndarray, thr: float = 0.5) -> np.ndarray:
	return (x > thr).astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_size: int = 50) -> np.ndarray:
	# mask: uint8 {0,1}
	num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
	out = np.zeros_like(mask, dtype=np.uint8)
	for i in range(1, num_labels):
		area = stats[i, cv2.CC_STAT_AREA]
		if area >= min_size:
			out[labels == i] = 1
	return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
	"""Fill small holes in a binary mask using morphological closing."""
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
	closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
	return closed


def collect_image_paths(images_dir: str) -> List[str]:
	"""Collect image paths once and deduplicate for case-insensitive filesystems."""
	valid_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
	seen = set()
	paths: List[str] = []
	for root, _, files in os.walk(images_dir):
		for fname in files:
			ext = os.path.splitext(fname)[1].lower()
			if ext not in valid_ext:
				continue
			full_path = os.path.join(root, fname)
			key = os.path.normcase(os.path.realpath(full_path))
			if key in seen:
				continue
			seen.add(key)
			paths.append(full_path)
	return sorted(paths)


def load_model_from_ckpt(model_path: str, device_t: torch.device, base_ch: int = 32, backbone: Optional[str] = None):
	state = torch.load(model_path, map_location=device_t, weights_only=False)
	meta = None
	state_dict = None
	if isinstance(state, dict) and "model" in state:
		state_dict = state["model"]
		meta = state.get("meta", None)
	else:
		state_dict = state

	if meta and backbone is None:
		backbone = meta.get("backbone", None)
	if backbone and backbone != "cnn":
		model = AttentionUNetTransformer(in_ch=1, out_ch=3, backbone=backbone, pretrained=False)
	else:
		model = AttentionUNet(in_ch=1, base_ch=base_ch, out_ch=3)
	model.load_state_dict(state_dict)
	model.to(device_t)
	model.eval()
	return model, meta


def predict_with_tta(model: torch.nn.Module, inp: torch.Tensor, use_tta: bool) -> torch.Tensor:
	"""Standard 4-augmentation TTA (identity + flips)."""
	if not use_tta:
		with torch.no_grad():
			return model(inp)
	logits_list: List[torch.Tensor] = []
	with torch.no_grad():
		logits_list.append(model(inp))
		inp_h = torch.flip(inp, dims=[3])
		log_h = model(inp_h)
		log_h = torch.flip(log_h, dims=[3])
		logits_list.append(log_h)
		inp_v = torch.flip(inp, dims=[2])
		log_v = model(inp_v)
		log_v = torch.flip(log_v, dims=[2])
		logits_list.append(log_v)
		inp_hv = torch.flip(inp, dims=[2, 3])
		log_hv = model(inp_hv)
		log_hv = torch.flip(log_hv, dims=[2, 3])
		logits_list.append(log_hv)
	return torch.stack(logits_list, dim=0).mean(dim=0)


def predict_with_d4_tta(model: torch.nn.Module, inp: torch.Tensor) -> torch.Tensor:
	"""Full D4 symmetry group TTA: 8 augmentations (4 rotations × 2 flips).

	This is the complete dihedral group of the square:
	identity, 90°, 180°, 270° rotations, and each with a horizontal flip.
	Particularly useful for retinal flatmounts with no canonical orientation.
	"""
	logits_list: List[torch.Tensor] = []
	with torch.no_grad():
		for k in range(4):  # 0°, 90°, 180°, 270°
			rotated = torch.rot90(inp, k=k, dims=[2, 3])
			# Forward on rotated
			logits = model(rotated)
			# Undo rotation
			logits = torch.rot90(logits, k=-k, dims=[2, 3])
			logits_list.append(logits)

			# With horizontal flip
			flipped = torch.flip(rotated, dims=[3])
			logits_f = model(flipped)
			logits_f = torch.flip(logits_f, dims=[3])
			logits_f = torch.rot90(logits_f, k=-k, dims=[2, 3])
			logits_list.append(logits_f)

	return torch.stack(logits_list, dim=0).mean(dim=0)


def load_ensemble_models(
	model_paths: List[str],
	threshold_paths: List[str],
	device_t: torch.device,
	backbone: str = "convnext_tiny",
) -> Tuple[List[torch.nn.Module], List[List[float]]]:
	"""Load multiple models for ensemble inference."""
	models = []
	thresholds_list = []
	for i, (mp, tp) in enumerate(zip(model_paths, threshold_paths)):
		print(f"  Loading model {i+1}/{len(model_paths)}: {os.path.basename(os.path.dirname(mp))}")
		model, meta = load_model_from_ckpt(mp, device_t, backbone=backbone)
		models.append(model)

		# Load thresholds
		thr = [0.5, 0.5, 0.5]
		if os.path.exists(tp):
			with open(tp, "r") as f:
				obj = json.load(f)
				thr = [float(obj.get("retina", 0.5)),
				       float(obj.get("nv", 0.5)),
				       float(obj.get("vo", 0.5))]
		thresholds_list.append(thr)
	return models, thresholds_list


def ensemble_predict(
	models: List[torch.nn.Module],
	inp: torch.Tensor,
	use_d4_tta: bool = True,
) -> torch.Tensor:
	"""Run ensemble prediction: average logits across all models with TTA."""
	all_logits: List[torch.Tensor] = []
	for model in models:
		if use_d4_tta:
			logits = predict_with_d4_tta(model, inp)
		else:
			logits = predict_with_tta(model, inp, use_tta=True)
		all_logits.append(logits)
	# Average logits (pre-sigmoid) across models
	return torch.stack(all_logits, dim=0).mean(dim=0)


def run_folder(
	model_path: str,
	images_dir: str,
	output_dir: str,
	base_ch: int = 32,
	device: str = "cuda",
	backbone: Optional[str] = None,
	tta: bool = True,
	min_component: int = 0,
	thresholds_json: Optional[str] = None,
):
	device_t = resolve_device(device)
	model, meta = load_model_from_ckpt(model_path, device_t, base_ch=base_ch, backbone=backbone)

	# thresholds
	thr = [0.5, 0.5, 0.5]
	if thresholds_json and os.path.exists(thresholds_json):
		with open(thresholds_json, "r") as f:
			obj = json.load(f)
			thr = [float(obj.get("retina", 0.5)), float(obj.get("nv", 0.5)), float(obj.get("vo", 0.5))]
	elif meta and isinstance(meta, dict) and "thresholds" in meta:
		vals = meta["thresholds"]
		if isinstance(vals, list) and len(vals) == 3:
			thr = [float(vals[0]), float(vals[1]), float(vals[2])]

	# Create category-based output directories
	os.makedirs(output_dir, exist_ok=True)
	dirs = {
		"tr_masks": os.path.join(output_dir, "TR masks"),
		"tr_overlays": os.path.join(output_dir, "TR overlays"),
		"ivnv_masks": os.path.join(output_dir, "IVNV masks"),
		"ivnv_overlays": os.path.join(output_dir, "IVNV overlays"),
		"ava_masks": os.path.join(output_dir, "AVA masks"),
		"ava_overlays": os.path.join(output_dir, "AVA overlays"),
		"originals": os.path.join(output_dir, "originals"),
	}
	for d in dirs.values():
		os.makedirs(d, exist_ok=True)

	records = []
	img_paths = []
	for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp", "*.PNG", "*.JPG", "*.JPEG", "*.TIF", "*.TIFF", "*.BMP"]:
		img_paths += glob.glob(os.path.join(images_dir, "**", ext), recursive=True)

	for p in img_paths:
		name = os.path.splitext(os.path.basename(p))[0]
		gray = _read_image_gray_any(p)
		H0, W0 = gray.shape[:2]
		gray_resized = _resize_gray(gray, (512, 512))
		inp = torch.from_numpy(gray_resized[None, None, ...].astype(np.float32))
		inp = (inp / 255.0 - 0.5) / 0.5
		inp = inp.to(device_t)

		logits = predict_with_tta(model, inp, use_tta=tta)
		probs = torch.sigmoid(logits)[0].cpu().numpy()

		mr512 = binarize(probs[0], thr=thr[0])
		mnv512 = binarize(probs[1], thr=thr[1])
		mvo512 = binarize(probs[2], thr=thr[2])

		# Upsample to original resolution
		mr = cv2.resize(mr512, (W0, H0), interpolation=cv2.INTER_NEAREST)
		mnv = cv2.resize(mnv512, (W0, H0), interpolation=cv2.INTER_NEAREST)
		mvo = cv2.resize(mvo512, (W0, H0), interpolation=cv2.INTER_NEAREST)

		# Constrain NV/VO to be within predicted retina
		mnv = (mnv & mr).astype(np.uint8)
		mvo = (mvo & mr).astype(np.uint8)

		# Optional post-processing
		if min_component and min_component > 0:
			mnv = remove_small_components(mnv, min_size=min_component)
			mvo = remove_small_components(mvo, min_size=min_component)

		# Save original image
		if save_originals and "originals" in dirs:
			import shutil
			shutil.copy2(p, os.path.join(dirs["originals"], os.path.basename(p)))

		# Write masks into category folders
		cv2.imwrite(os.path.join(dirs["tr_masks"], f"{name}.png"), (mr * 255).astype(np.uint8))
		cv2.imwrite(os.path.join(dirs["ivnv_masks"], f"{name}.png"), (mnv * 255).astype(np.uint8))
		cv2.imwrite(os.path.join(dirs["ava_masks"], f"{name}.png"), (mvo * 255).astype(np.uint8))

		# Write overlays into category folders
		over_retina = overlay_mask(gray, mr, (255, 255, 255), 0.25)
		over_nv = overlay_mask(gray, mnv, (255, 0, 0), 0.5)
		over_vo = overlay_mask(gray, mvo, (255, 255, 0), 0.5)
		cv2.imwrite(os.path.join(dirs["tr_overlays"], f"{name}.png"), over_retina)
		cv2.imwrite(os.path.join(dirs["ivnv_overlays"], f"{name}.png"), over_nv)
		cv2.imwrite(os.path.join(dirs["ava_overlays"], f"{name}.png"), over_vo)

		# areas and ratios (in original resolution pixels)
		area_retina = int(mr.sum())
		area_nv = int(mnv.sum())
		area_vo = int(mvo.sum())
		ratio_nv = (area_nv / area_retina * 100.0) if area_retina > 0 else 0.0
		ratio_vo = (area_vo / area_retina * 100.0) if area_retina > 0 else 0.0

		records.append({
			"file": name,
			"retina_area": area_retina,
			"ivnv_area": area_nv,
			"ava_area": area_vo,
			"ivnv_pct_of_retina": ratio_nv,
			"ava_pct_of_retina": ratio_vo,
		})

	df = pd.DataFrame.from_records(records)
	df.to_excel(os.path.join(output_dir, "metrics.xlsx"), index=False)
	print(f"Saved results to {output_dir}")


def run_ensemble_folder(
	ensemble_dir: str,
	images_dir: str,
	output_dir: str,
	device: str = "cpu",
	use_d4_tta: bool = True,
	min_component: int = 50,
	ava_closing: bool = True,
	backbone: str = "convnext_tiny",
	save_tr_masks: bool = True,
	save_tr_overlays: bool = True,
	save_ivnv_masks: bool = True,
	save_ivnv_overlays: bool = True,
	save_ava_masks: bool = True,
	save_ava_overlays: bool = True,
	save_metrics: bool = True,
	save_originals: bool = False,
):
	"""Run ensemble inference with all fold models.

	Discovers models from ensemble_dir/fold_*/best.pth (or best_tr.pth for legacy).
	Uses D4 TTA (8 augmentations) and averages logits across all models.
	"""
	device_t = resolve_device(device)

	# Discover fold models
	model_paths = []
	threshold_paths = []

	for fold_idx in range(5):
		fold_dir = os.path.join(ensemble_dir, f"fold_{fold_idx}")
		# Check for model files in priority order
		candidates = ["best.pth", "best_tr.pth", "model.pth"]
		found = False
		for cand in candidates:
			cand_path = os.path.join(fold_dir, cand)
			if os.path.exists(cand_path):
				model_paths.append(cand_path)
				thr_path = os.path.join(fold_dir, "thresholds.json")
				threshold_paths.append(thr_path)
				found = True
				break
		if not found:
			print(f"WARNING: No model found for fold {fold_idx} in {fold_dir}")

	print(f"Loading {len(model_paths)} models for ensemble...")
	if not model_paths:
		raise RuntimeError(f"No fold models found in ensemble directory: {ensemble_dir}")
	models, thresholds_list = load_ensemble_models(model_paths, threshold_paths, device_t, backbone=backbone)

	# Average thresholds across folds for ensemble
	avg_thr = np.mean(thresholds_list, axis=0).tolist()
	print(f"Ensemble thresholds (averaged): R={avg_thr[0]:.2f}, NV={avg_thr[1]:.2f}, VO={avg_thr[2]:.2f}")

	# Create output directories
	os.makedirs(output_dir, exist_ok=True)
	dirs: Dict[str, str] = {}
	if save_tr_masks:
		dirs["tr_masks"] = os.path.join(output_dir, "TR masks")
	if save_tr_overlays:
		dirs["tr_overlays"] = os.path.join(output_dir, "TR overlays")
	if save_ivnv_masks:
		dirs["ivnv_masks"] = os.path.join(output_dir, "IVNV masks")
	if save_ivnv_overlays:
		dirs["ivnv_overlays"] = os.path.join(output_dir, "IVNV overlays")
	if save_ava_masks:
		dirs["ava_masks"] = os.path.join(output_dir, "AVA masks")
	if save_ava_overlays:
		dirs["ava_overlays"] = os.path.join(output_dir, "AVA overlays")
	if save_originals:
		dirs["originals"] = os.path.join(output_dir, "originals")

	for out_dir in dirs.values():
		os.makedirs(out_dir, exist_ok=True)

	# Find images
	img_paths = collect_image_paths(images_dir)

	records = []
	tta_mode = "D4 (8-aug)" if use_d4_tta else "standard (4-aug)"
	total_passes = len(models) * (8 if use_d4_tta else 4)
	print(f"Processing {len(img_paths)} images with {len(models)}-model ensemble, {tta_mode} TTA ({total_passes} forward passes/image)")

	for i, p in enumerate(img_paths):
		name = os.path.splitext(os.path.basename(p))[0]
		print(f"  [{i+1}/{len(img_paths)}] {name}")

		gray = _read_image_gray_any(p)
		H0, W0 = gray.shape[:2]
		gray_resized = _resize_gray(gray, (512, 512))
		inp = torch.from_numpy(gray_resized[None, None, ...].astype(np.float32))
		inp = (inp / 255.0 - 0.5) / 0.5
		inp = inp.to(device_t)

		# Ensemble prediction (averaged logits)
		logits = ensemble_predict(models, inp, use_d4_tta=use_d4_tta)
		probs = torch.sigmoid(logits)[0].cpu().numpy()

		# Binarize with averaged thresholds
		mr512 = binarize(probs[0], thr=avg_thr[0])
		mnv512 = binarize(probs[1], thr=avg_thr[1])
		mvo512 = binarize(probs[2], thr=avg_thr[2])

		# Upsample to original resolution
		mr = cv2.resize(mr512, (W0, H0), interpolation=cv2.INTER_NEAREST)
		mnv = cv2.resize(mnv512, (W0, H0), interpolation=cv2.INTER_NEAREST)
		mvo = cv2.resize(mvo512, (W0, H0), interpolation=cv2.INTER_NEAREST)

		# Post-processing: fill holes in TR first
		mr = fill_holes(mr)

		# Constrain NV/VO to be within predicted retina
		mnv = (mnv & mr).astype(np.uint8)
		mvo = (mvo & mr).astype(np.uint8)

		# AVA morphological closing to fill gaps
		if ava_closing:
			mvo = fill_holes(mvo)
			# Re-constrain after closing
			mvo = (mvo & mr).astype(np.uint8)

		# Remove small components
		if min_component > 0:
			mnv = remove_small_components(mnv, min_size=min_component)
			mvo = remove_small_components(mvo, min_size=min_component)

		# Save original image
		import shutil
		shutil.copy2(p, os.path.join(dirs["originals"], os.path.basename(p)))

		# Write masks
		if save_tr_masks and "tr_masks" in dirs:
			cv2.imwrite(os.path.join(dirs["tr_masks"], f"{name}.png"), (mr * 255).astype(np.uint8))
		if save_ivnv_masks and "ivnv_masks" in dirs:
			cv2.imwrite(os.path.join(dirs["ivnv_masks"], f"{name}.png"), (mnv * 255).astype(np.uint8))
		if save_ava_masks and "ava_masks" in dirs:
			cv2.imwrite(os.path.join(dirs["ava_masks"], f"{name}.png"), (mvo * 255).astype(np.uint8))

		# Write overlays
		if save_tr_overlays and "tr_overlays" in dirs:
			over_retina = overlay_mask(gray, mr, (255, 255, 255), 0.25)
			cv2.imwrite(os.path.join(dirs["tr_overlays"], f"{name}.png"), over_retina)
		if save_ivnv_overlays and "ivnv_overlays" in dirs:
			over_nv = overlay_mask(gray, mnv, (255, 0, 0), 0.5)
			cv2.imwrite(os.path.join(dirs["ivnv_overlays"], f"{name}.png"), over_nv)
		if save_ava_overlays and "ava_overlays" in dirs:
			over_vo = overlay_mask(gray, mvo, (255, 255, 0), 0.5)
			cv2.imwrite(os.path.join(dirs["ava_overlays"], f"{name}.png"), over_vo)

		# Metrics
		area_retina = int(mr.sum())
		area_nv = int(mnv.sum())
		area_vo = int(mvo.sum())
		ratio_nv = (area_nv / area_retina * 100.0) if area_retina > 0 else 0.0
		ratio_vo = (area_vo / area_retina * 100.0) if area_retina > 0 else 0.0

		if save_metrics:
			records.append({
				"file": name,
				"retina_area": area_retina,
				"ivnv_area": area_nv,
				"ava_area": area_vo,
				"ivnv_pct_of_retina": ratio_nv,
				"ava_pct_of_retina": ratio_vo,
			})

	if save_metrics:
		df = pd.DataFrame.from_records(records)
		df.to_excel(os.path.join(output_dir, "metrics.xlsx"), index=False)
	print(f"\nEnsemble inference complete. Saved to {output_dir}")


if __name__ == "__main__":
	import argparse
	default_ensemble_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "weights"))

	parser = argparse.ArgumentParser(
		description="OIR Retinal Segmentation Inference. Default: 5-model ensemble with D4 TTA and AVA closing."
	)
	parser.add_argument("--images", type=str, required=True, help="Input image directory")
	parser.add_argument("--out", type=str, required=True, help="Output directory")
	parser.add_argument("--min_component", type=int, default=50, help="Min connected component size (default: 50)")
	parser.add_argument("--backbone", type=str, default="convnext_tiny", help="Backbone architecture (default: convnext_tiny)")
	parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Inference device selection")

	# Ensemble mode (DEFAULT)
	parser.add_argument("--ensemble_dir", type=str, default=default_ensemble_dir,
	                    help="Directory with fold model weights")
	parser.add_argument("--no_d4_tta", action="store_true", help="Disable D4 TTA (use standard 4-aug TTA instead)")
	parser.add_argument("--no_ava_closing", action="store_true", help="Disable morphological closing on AVA masks")

	# Single-model mode (legacy, must explicitly opt in)
	parser.add_argument("--single_model", type=str, default="",
	                    help="Path to a single model checkpoint (disables ensemble, uses legacy single-model inference)")
	parser.add_argument("--base_ch", type=int, default=32, help="Base channels for legacy AttentionUNet (only with --single_model)")
	parser.add_argument("--tta", action="store_true", help="Enable TTA for single-model mode")
	parser.add_argument("--thresholds", type=str, default="", help="Thresholds JSON for single-model mode")

	args = parser.parse_args()

	if args.single_model:
		# Legacy single-model inference
		print("⚠ Running in SINGLE-MODEL mode (legacy). Use ensemble (default) for best results.")
		run_folder(
			model_path=args.single_model,
			images_dir=args.images,
			output_dir=args.out,
			base_ch=args.base_ch,
			backbone=(args.backbone if args.backbone != "convnext_tiny" else None),
			tta=args.tta,
			min_component=args.min_component,
			thresholds_json=(args.thresholds if args.thresholds else None),
			device=args.device,
		)
	else:
		# Default: ensemble inference with D4 TTA + AVA closing
		run_ensemble_folder(
			ensemble_dir=args.ensemble_dir,
			images_dir=args.images,
			output_dir=args.out,
			device=args.device,
			use_d4_tta=(not args.no_d4_tta),
			min_component=args.min_component,
			ava_closing=(not args.no_ava_closing),
			backbone=args.backbone,
		)
