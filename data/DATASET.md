- Source used: Hugging Face `TSY-0408/PlantVillage` (loader script; data files fetched from
  `mohanty/PlantVillage`), config `default` (= color).

  ## Known limitation: leaf grouping coverage
The split is grouped by the dataset's `leaf-map.json`. About 39.8% of the selected images (all
corn classes, tomato mosaic virus and tomato target spot, plus 37-52% of four other tomato
classes) are not in that map. They are grouped by filename only, so photos of one physical leaf
could still be spread across splits. Lab metrics for those classes are therefore upper bounds.
The manifests carry `leaf_id_source` (`leaf_map` or `fallback`); evaluation reports the leaf-verified
subset separately. The field test set is the unbiased measurement.