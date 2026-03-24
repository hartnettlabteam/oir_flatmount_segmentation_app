# OIR Flatmount Segmentation App

Desktop app for OIR flatmount segmentation (TR, IVNV, AVA).

This repository is intended for end users who want a simple GUI workflow:

1. select an input folder,
2. select an output folder,
3. run segmentation.

No model-path setup is required in the GUI. Release builds bundle all required ensemble weights.

## Download

Get the latest release assets from GitHub Releases:

- Windows: `OIR-Flatmount-Segmentation-Windows.zip`
- macOS: `OIR-Flatmount-Segmentation-macOS-app.zip` (unsigned)

## Windows Quick Start (recommended)

1. Download `OIR-Flatmount-Segmentation-Windows.zip`.
2. Extract the zip.
3. Open the extracted folder and run `OIR Flatmount Segmentation.exe`.
4. In the app:
   - choose input folder,
   - choose output folder,
   - select desired outputs (masks, overlays, metrics, originals),
   - click `Run`.

## macOS Quick Start (unsigned app)

Because this app is unsigned (no Apple Developer notarization), Gatekeeper may warn on first launch.

1. Download `OIR-Flatmount-Segmentation-macOS-app.zip`.
2. Extract and move `OIR Flatmount Segmentation.app` to `Applications` (optional).
3. First launch:
   - right-click the app -> `Open` -> `Open`, or
   - go to `System Settings` -> `Privacy & Security` -> `Open Anyway`.
4. If macOS still blocks launch, run:

```bash
xattr -dr com.apple.quarantine "/Applications/OIR Flatmount Segmentation.app"
```

Then open again.

## macOS Fallback (no Apple license required)

If the unsigned app is blocked on your macOS version, run from source:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --no-compile -r requirements.txt
python app/gui.py
```

This does not require an Apple Developer account.

## Output files

Depending on selected checkboxes in the GUI, outputs include:

- `TR masks`
- `TR overlays`
- `IVNV masks`
- `IVNV overlays`
- `AVA masks`
- `AVA overlays`
- `metrics.xlsx` (with `ivnv_area`, `ava_area`, and percentage columns)
- `originals`
