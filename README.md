# AgriScan

Upload a photo of a tomato, potato, pepper, corn, rice or eggplant (brinjal) leaf. AgriScan names the most likely disease,
explains why it happens and what a grower can do, in **Bangla or English**. When the model is not
confident enough it says so instead of guessing.

**Live demo:** _add the link here after deploying_ (free host: the first visit after a quiet period takes about a minute to wake up)

> A student portfolio project, not agronomic advice. Check with your Upazila Agriculture Office
> before spraying or removing a crop.

## How it works

- **Model:** ResNet18 fine-tuned for 31 classes (12 conditions of rice and eggplant, 19 of tomato,
  potato, pepper and corn), exported to ONNX (about 43 MB) and served on CPU with ONNX Runtime
  (about 25 ms per photo for the model alone). Calibrated with temperature scaling. A
  MobileNetV3-Small was trained on the same data and scored a little lower, so it was not used.
- **Training data:** PlantVillage lab photos (the 19 older classes), plus real-world photos: PlantDoc
  for the older crops, and the Dhan-Shomadhan (rice) and Brinjal-BD (eggplant) datasets for the new
  ones. Rice and eggplant have no lab photos and no healthy rice class. Photos were split by leaf
  group so shots of one leaf never sit on both sides of the train/test split; exact and
  near-duplicate photos were removed.
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

Version 2 (ResNet18, 31 classes), measured on photos the model never trained on:

| Test set | Photos | Right answer |
|---|---|---|
| PlantVillage lab photos (19 older classes) | 3,991 | 98.4% |
| Real-world photos, all crops together | 711 | 70.5% (72.9% if the crop is picked first) |
| of which PlantDoc (older crops) | 134 | 54.5% |
| of which rice (Dhan-Shomadhan) | 244 | 65.6% |
| of which eggplant (Brinjal-BD) | 333 | 80.5% |

For comparison, version 1 (19 classes) got about 49% on the PlantDoc photos; adding rice and
eggplant did not make the older crops worse. With the "not sure" rule on, the model answers about
55% of real-world photos and is right on 84% of those it answers; it also answers about 11% of
photos of other plants' leaves instead of saying "not sure". Full tables are in `reports/`.

Limits to keep in mind:

- Real-world test sets are small, so numbers carry wide uncertainty (the 95% interval for the
  711-photo figure is 67-74%). Five classes have too few real-world test photos to score.
- Rice and eggplant test photos come from the same datasets and photographers as their training
  photos. Leaf groups are kept apart, but a stranger's phone photo will likely score lower.
- Rice leaf scald (about 25% recall) and rice brown spot (64%) are often mixed up with other rice
  diseases. Eggplant Phytophthora blight and powdery mildew have only 51 and 71 photos in total.
- PlantDoc labels were made by its authors and a few class mappings are approximate.
- The model knows only 31 conditions of six crops. It was tested on other plants' leaves, not on
  screenshots, hands or soil.
- Lab accuracy is not field accuracy: about half of real-world photos still get a wrong or
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
- Dhan-Shomadhan (rice leaf diseases) and Brinjal-BD (eggplant leaf diseases) datasets, both
  CC BY 4.0. _Add the full citations from each dataset's page here._
- Advice drawn from the extension publications listed under each result.
