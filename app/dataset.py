import os
import glob
from typing import Dict, List, Optional, Tuple

import cv2
import warnings
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None  # allow very large images
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
import numpy as np
from skimage import io as skio
from skimage.exposure import rescale_intensity
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MASK_EXTS = IMAGE_EXTS


def _read_image_gray_any(path: str) -> np.ndarray:
	# Robust reader for 8/16-bit PNG/JPG/TIFF. Returns uint8 grayscale [0..255]
	ext = os.path.splitext(path)[1].lower()
	img = None
	try:
		arr = skio.imread(path)
		if arr.ndim == 2:
			img = arr
		elif arr.ndim == 3:
			# Choose the strongest channel akin to Lua pipeline
			ch_means = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
			ch = int(np.argmax(ch_means))
			img = arr[..., ch]
		else:
			raise ValueError("Unsupported image shape")
	except Exception:
		# Fallback to OpenCV
		bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
		if bgr is None:
			raise
		if bgr.ndim == 2:
			img = bgr
		elif bgr.ndim == 3:
			means = bgr.reshape(-1, bgr.shape[-1]).mean(axis=0)
			ch = int(np.argmax(means))
			img = bgr[..., ch]
		else:
			raise ValueError("Unsupported image shape")

	# Normalize to uint8
	if img.dtype == np.uint16:
		img = rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)
	elif img.dtype != np.uint8:
		img = rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)
	return img


def _read_mask_binary(path: str, out_size: Tuple[int, int]) -> np.ndarray:
	mask = _read_image_gray_any(path)
	mask = (mask > 127).astype(np.uint8)
	mask = cv2.resize(mask, out_size, interpolation=cv2.INTER_NEAREST)
	return mask


def _resize_gray(img: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
	if img.ndim != 2:
		raise ValueError("Expected grayscale image")
	return cv2.resize(img, out_size, interpolation=cv2.INTER_AREA)


def _index_masks(mask_dir: str) -> Dict[str, str]:
	index: Dict[str, str] = {}
	if not mask_dir or not os.path.isdir(mask_dir):
		return index
	for ext in MASK_EXTS:
		for p in glob.glob(os.path.join(mask_dir, f"**/*{ext}"), recursive=True):
			base = os.path.splitext(os.path.basename(p))[0]
			index[base.lower()] = p
	return index


class OIRSegmentationDataset:
	def __init__(
		self,
		images_dir: str,
		masks_retina_dir: str,
		masks_nv_dir: str,
		masks_vo_dir: str,
		image_size: int = 512,
		augment: bool = True,
		strong_augment: bool = False,
		files: Optional[List[str]] = None,
		require_nv: bool = True,
		require_vo: bool = True,
		require_retina: bool = False,
	):
		self.images_dir = images_dir
		self.mr_dir = masks_retina_dir
		self.mnv_dir = masks_nv_dir
		self.mvo_dir = masks_vo_dir
		self.size = image_size
		self.augment_flag = augment
		self.strong_augment = strong_augment
		self.require_nv = require_nv
		self.require_vo = require_vo
		self.require_retina = require_retina

		self.retina_idx = _index_masks(self.mr_dir)
		self.nv_idx = _index_masks(self.mnv_dir)
		self.vo_idx = _index_masks(self.mvo_dir)

		if files is None:
			self.files = []
			for ext in IMAGE_EXTS:
				self.files += glob.glob(os.path.join(self.images_dir, f"**/*{ext}"), recursive=True)
			self.files.sort()
		else:
			self.files = files

		# filter based on required masks (optionally require retina)
		self.samples: List[str] = []
		for p in self.files:
			base = os.path.splitext(os.path.basename(p))[0].lower()
			have_nv = base in self.nv_idx
			have_vo = base in self.vo_idx
			have_ret = base in self.retina_idx
			if ((not self.require_nv) or have_nv) and ((not self.require_vo) or have_vo) and ((not self.require_retina) or have_ret):
				self.samples.append(p)

		basic_aug = [
			A.HorizontalFlip(p=0.5),
			A.VerticalFlip(p=0.5),
			A.Rotate(limit=180, p=0.75, border_mode=cv2.BORDER_REFLECT_101),
			A.RandomBrightnessContrast(p=0.5),
			A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
			A.GaussNoise(p=0.2),
		]
		strong_aug = [
			A.OneOf([
				A.ElasticTransform(alpha=50, sigma=6, p=1.0),
				A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
				A.OpticalDistortion(distort_limit=0.3, p=1.0),
			], p=0.3),
			A.CoarseDropout(p=0.3),
			A.MotionBlur(p=0.2),
			A.RandomGamma(p=0.2),
		]
		self.transform_train = A.Compose(
			(basic_aug + (strong_aug if self.strong_augment else [])),
			additional_targets={
				"mask_retina": "mask",
				"mask_nv": "mask",
				"mask_vo": "mask",
			},
		)
		self.transform_to_tensor = A.Compose(
			[
				A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
				ToTensorV2(),
			]
		)

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int):
		img_path = self.samples[idx]
		base = os.path.splitext(os.path.basename(img_path))[0]
		img = _read_image_gray_any(img_path)
		img = _resize_gray(img, (self.size, self.size))

		# Retina optional
		has_retina = base.lower() in self.retina_idx
		if has_retina:
			mr = _read_mask_binary(self.retina_idx[base.lower()], (self.size, self.size))
		else:
			mr = np.zeros((self.size, self.size), dtype=np.uint8)
		if base.lower() in self.nv_idx:
			mnv = _read_mask_binary(self.nv_idx[base.lower()], (self.size, self.size))
		else:
			mnv = np.zeros((self.size, self.size), dtype=np.uint8)
		if base.lower() in self.vo_idx:
			mvo = _read_mask_binary(self.vo_idx[base.lower()], (self.size, self.size))
		else:
			mvo = np.zeros((self.size, self.size), dtype=np.uint8)

		if self.augment_flag:
			aug = self.transform_train(image=img, mask_retina=mr, mask_nv=mnv, mask_vo=mvo)
			img = aug["image"]
			mr = aug["mask_retina"]
			mnv = aug["mask_nv"]
			mvo = aug["mask_vo"]

		# to tensor
		img_tensor = self.transform_to_tensor(image=img)["image"]  # [1,H,W] float32
		mr_tensor = (mr > 0).astype(np.float32)
		mnv_tensor = (mnv > 0).astype(np.float32)
		mvo_tensor = (mvo > 0).astype(np.float32)

		# Stack masks into channels [3,H,W]
		mask_stack = np.stack([mr_tensor, mnv_tensor, mvo_tensor], axis=0)

		return {
			"image": img_tensor,
			"masks": mask_stack,
			"name": base,
			"path": img_path,
			"has_retina": has_retina,
		}


class OIRSegmentationNVCropsDataset(OIRSegmentationDataset):
	def __init__(
		self,
		images_dir: str,
		masks_retina_dir: str,
		masks_nv_dir: str,
		masks_vo_dir: str,
		image_size: int = 512,
		augment: bool = True,
		files: Optional[List[str]] = None,
		crop_size: int = 384,
		crops_per_image: int = 2,
		require_nv: bool = True,
		require_vo: bool = True,
		require_retina: bool = False,
	):
		super().__init__(images_dir, masks_retina_dir, masks_nv_dir, masks_vo_dir, image_size, augment, False, files, require_nv=require_nv, require_vo=require_vo, require_retina=require_retina)
		self.crop_size = crop_size
		self.crops_per_image = crops_per_image
		# Build expanded index
		self.expanded: List[Tuple[str, int]] = []
		for p in self.samples:
			for k in range(self.crops_per_image):
				self.expanded.append((p, k))

	def __len__(self) -> int:
		return len(self.expanded)

	def __getitem__(self, idx: int):
		img_path, _ = self.expanded[idx]
		base = os.path.splitext(os.path.basename(img_path))[0]
		img = _read_image_gray_any(img_path)
		H, W = img.shape[:2]
		# read masks at original size first for crop
		has_retina = base.lower() in self.retina_idx
		if has_retina:
			mr_full = _read_image_gray_any(self.retina_idx[base.lower()])
			mr_full = (mr_full > 127).astype(np.uint8)
		else:
			mr_full = np.zeros((H, W), dtype=np.uint8)
		if base.lower() in self.nv_idx:
			mnv_full = _read_image_gray_any(self.nv_idx[base.lower()])
			mnv_full = (mnv_full > 127).astype(np.uint8)
		else:
			mnv_full = np.zeros((H, W), dtype=np.uint8)
		if base.lower() in self.vo_idx:
			mvo_full = _read_image_gray_any(self.vo_idx[base.lower()])
			mvo_full = (mvo_full > 127).astype(np.uint8)
		else:
			mvo_full = np.zeros((H, W), dtype=np.uint8)

		# choose NV-positive center if available
		ys, xs = np.where(mnv_full > 0)
		if ys.size > 0:
			cid = np.random.randint(0, ys.size)
			cy, cx = int(ys[cid]), int(xs[cid])
		else:
			cy, cx = H // 2, W // 2

		L = self.crop_size
		y0 = max(0, cy - L // 2)
		x0 = max(0, cx - L // 2)
		y1 = min(H, y0 + L)
		x1 = min(W, x0 + L)
		y0 = max(0, y1 - L)
		x0 = max(0, x1 - L)

		img_c = img[y0:y1, x0:x1]
		mr_c = mr_full[y0:y1, x0:x1]
		mnv_c = mnv_full[y0:y1, x0:x1]
		mvo_c = mvo_full[y0:y1, x0:x1]

		# resize crop back to model input size
		img = _resize_gray(img_c, (self.size, self.size))
		mr = cv2.resize(mr_c, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
		mnv = cv2.resize(mnv_c, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
		mvo = cv2.resize(mvo_c, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

		if self.augment_flag:
			aug = self.transform_train(image=img, mask_retina=mr, mask_nv=mnv, mask_vo=mvo)
			img = aug["image"]
			mr = aug["mask_retina"]
			mnv = aug["mask_nv"]
			mvo = aug["mask_vo"]

		img_tensor = self.transform_to_tensor(image=img)["image"]
		mask_stack = np.stack([(mr > 0).astype(np.float32), (mnv > 0).astype(np.float32), (mvo > 0).astype(np.float32)], axis=0)
		return {
			"image": img_tensor,
			"masks": mask_stack,
			"name": base + "_crop",
			"path": img_path,
			"has_retina": has_retina,
		}
