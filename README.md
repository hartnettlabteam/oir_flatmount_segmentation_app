# OIR Flatmount Segmentation App

Desktop app for OIR flatmount segmentation (TR, IVNV, AVA).

This repository is intended for end users who want a simple GUI workflow:

1. select an input folder or a single image,
2. select an output folder,
3. run segmentation.

## Download

- Windows installer (recommended): `OIR-Flatmount-Segmentation-Windows-Setup.exe`
- Windows portable zip: `OIR-Flatmount-Segmentation-Windows.zip`
- macOS: `OIR-Flatmount-Segmentation-macOS-app.zip` (unsigned)

## Windows Quick Start (recommended — installer)

1. Download `OIR-Flatmount-Segmentation-Windows-Setup.exe`. You can find this in the [Releases.](https://github.com/hartnettlabteam/oir_flatmount_segmentation_app/releases)
2. Run it and step through the wizard. You can optionally create a Start Menu shortcut and a Desktop shortcut on the "Select Additional Tasks" page.
3. Launch the app from the Start Menu or Desktop shortcut.
4. In the app:
   - choose input folder or single image,
   - choose output folder,
   - select desired outputs (masks, overlays, metrics, originals),
   - click `Run`.

To remove the app later, use `Settings -> Apps` (or `Add or Remove Programs`) and uninstall "OIR Flatmount Segmentation".

## Windows Quick Start (portable zip)

If you cannot install software, use the portable zip instead:

1. Download `OIR-Flatmount-Segmentation-Windows.zip`.
2. Extract the zip anywhere.
3. Open the extracted folder and run `OIR Flatmount Segmentation.exe`.

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

## macOS Fallback 

If the unsigned app is blocked on your macOS version, run from source:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --no-compile -r requirements.txt
python app/gui.py
```

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

## Citation and Contact

If you use this model, please cite the associated Hartnett Lab [TVST manuscript](https://iovs.arvojournals.org/article.aspx?articleid=2817567)

"Shah, N.S., Ramshekar, A., Asare-Bediako, B., Tankersley, M., Randolph, J.W., Whitney, J. and Hartnett, M.E., 2026. AI-Assisted Quantification of Retinal Vascular Parameters in Oxygen-Induced Retinopathy Models: A U-Net Approach. Investigative Ophthalmology & Visual Science, 67(7), pp.5204-5204."

Contact:
- Neal Shah: neals1@stanford.edu
- Aniket Ramshekar: aniket.ramshekar@stanford.edu
- M. Elizabeth Hartnett: me.hartnett@stanford.edu

