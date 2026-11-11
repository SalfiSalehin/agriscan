
- Licence: UNVERIFIED. The Hugging Face page lists CC BY-SA 3.0, but the loader script itself marks
  that as an assumption. Check the original sources (Mendeley Data / Kaggle / the paper) and
  record the licence and the date checked here: ____


- Source used: Hugging Face `TSY-0408/PlantVillage` (loader script; data files fetched from
  `mohanty/PlantVillage`), config `default` (= color).

  ## Known limitation: leaf grouping coverage
The split is grouped by the dataset's `leaf-map.json`. About 39.8% of the selected images (all
corn classes, tomato mosaic virus and tomato target spot, plus 37-52% of four other tomato
classes) are not in that map. They are grouped by filename only, so photos of one physical leaf
could still be spread across splits. Lab metrics for those classes are therefore upper bounds.
The manifests carry `leaf_id_source` (`leaf_map` or `fallback`); evaluation reports the leaf-verified
subset separately. The field test set is the unbiased measurement.

## Leaf masks for the background augmentation
Cut-out leaf shapes are derived from PlantVillage's own segmented images with
`python -m ml.make_masks` and used only for the background-replacement augmentation.
Training, validation and test images are the colour originals. Switch it off with `--bg-prob 0`.