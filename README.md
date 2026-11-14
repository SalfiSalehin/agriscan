---
title: AgriScan
emoji: 🌿
colorFrom: green
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Leaf-disease hints for tomato, potato, pepper and corn (Bangla/English)
---

# AgriScan

Upload a photo of a tomato, potato, pepper or corn leaf. AgriScan names the most likely disease,
explains why it happens and what a grower can do, in **Bangla or English**. When the model is not
confident enough it says so instead of guessing.

**Live demo:** _add your Hugging Face Space link here after deploying_

> A student portfolio project, not agronomic advice. Check with your Upazila Agriculture Office
> before spraying or removing a crop.

## How it works

- **Model:** MobileNetV3-Small fine-tuned on PlantVillage (19 classes), exported to ONNX and served
  on CPU with ONNX Runtime. Calibrated with temperature scaling.
- **Real-world photos:** PlantVillage photos are studio-style, so the model was also trained on a
  share of real-world photos from PlantDoc. A separate, official PlantDoc test folder was never
  trained on; near-duplicates between its photos and the training photos were checked and removed.
- **"Not sure" rule:** if the model's calibrated confidence is below a threshold chosen on held-out
  photos, the page shows photo-taking tips instead of a diagnosis (and, if the model is at least
  50% sure, lets you read about its top guess, clearly marked as unconfirmed).
- **Advice:** summarised from university extension services (sources listed under every result).
  The text names no pesticide products or doses. The Bangla text was written by the author and
  needs review by a native speaker or an agriculture officer.
- **Backend:** FastAPI. Uploads are decoded in memory and never saved. Format is decided from the
  file's bytes, size and pixel limits apply, requests are rate limited, and the page runs under a
  strict Content-Security-Policy.
- **Feedback:** an optional correct/wrong button stores the model's answer, language and your
  verdict (never the photo, filename or IP address).

## Honest results

| | PlantVillage (lab photos) | Real-world photos (PlantDoc test) |
|---|---|---|
| Before mixing in real-world photos | very high | about 36% right (crop unknown) |
| After | about 1 point lower | about 49% right (crop unknown), about 61% (crop picked) |

With the "not sure" rule on, the model answered about 39% of real-world photos and was right on
roughly 73–76% of those it answered. Full tables are in `reports/`.

Limits to keep in mind:

- The real-world test set is small (134 photos, 14 classes scored), so the numbers have wide
  uncertainty. Some classes have no real-world test photos at all.
- PlantDoc labels were made by its authors and a few class mappings are approximate.
- The model knows only 19 conditions of four crops. It was not tested on non-leaf photos.
- Lab accuracy is not field accuracy: roughly half of real-world photos still get a wrong or
  "not sure" answer.

## Run it locally (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m pytest -q
uvicorn app.main:app --reload --port 8010   # then open http://127.0.0.1:8010/
```

Settings (environment variables, all optional): `AGRISCAN_RATE_PER_MIN` (default 30, 0 = off),
`AGRISCAN_MAX_UPLOAD_MB`, `AGRISCAN_DB` (path, or `off`), `AGRISCAN_DOCS` (`off` hides Swagger),
`AGRISCAN_TRUST_PROXY` (set to 1 behind a reverse proxy).

## Credits and data

- PlantVillage dataset (lab photos). Leaf masks derived from it were used to place leaves on
  varied backgrounds during training.
- PlantDoc dataset, CC BY 4.0: Singh et al., "PlantDoc: A Dataset for Visual Plant Disease
  Detection" (2020).
- Advice drawn from the extension publications listed under each result.
