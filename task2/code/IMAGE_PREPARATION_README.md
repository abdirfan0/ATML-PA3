# Task 2 selected images — CPU

Run `test_task2_images.py`, then:

```bash
python task2_prepare_images.py --task-dir /content/drive/MyDrive/ATML/PA3/task2 --workers 8 --work-dir /content/pa3_task2_image_work
```

Uses the saved 6,528-image catalog and validates its selection-manifest identity. Only the selected JPEGs are fetched from COCO's official image bucket. No full COCO image archives, feature extraction, model downloads, training or GPU calls are involved.

## Storage and recovery

The downloader works on local CPU disk and saves uncompressed TAR archives of 256 images to `data/images/` on Drive (26 archives, the last with 128 images). JPEGs are already compressed; stored source bytes are not resized or re-encoded. Every source image is checked for JPEG format, annotated dimensions, successful full RGB decoding and SHA256. The persistent archive copy is hash verified before a completion marker is saved. A failed/unavailable source stops preparation; it is never replaced or resampled.

Completed archives and their image records are reused after identity/hash checks. Following a runtime disconnect, only the unfinished archive needs downloading again. Local work files are disposable caches, while Drive archives/markers persist. Do not remove the original annotation files or split manifests. Temporary archive files without a completion marker can be rebuilt safely on rerun.

`data/images/manifest.json` maps each selected image to its archive member, original dimensions, byte count and SHA256. `ImageArchiveReader` provides RGB images directly from these archives, without requiring thousands of JPEG files on Drive. The feature-caching stage can copy the archives to local disk to keep Drive I/O out of the GPU loop.

`results/image_preparation.json` records verified sizes, timing for this session and completion status. `configs/image_preparation_config.json` binds the image archives to the exact selected catalog and split-manifest hash.

## Label inspection

The script creates 22 JPEG contact sheets in `figures/label_review/`, covering all 24 annotation-derived development pairs and all 64 annotation-derived final-test pairs, ordered by saved pair ID, four pairs per sheet. Each row shows the shared question, both images, and annotation targets/IDs. This selection rule uses no model predictions. Images are scaled only for the contact sheets, not for stored training inputs.

`results/label_review_index.json` lists sheet paths, hashes and covered pair IDs. Review status remains **pending human label review** until inspection. Presence negatives can contain unannotated objects; counting annotations may be ambiguous. Record such findings before training. Do not select or replace evaluation examples based on model performance. This preparation does not claim the labels have been visually verified merely because downloads/decode checks pass.

Eight CPU tests cover image decoding, dimension/format rejection, path validation, deterministic archives, RGB retrieval, completed-archive identity/hash checks, cached downloads, and failure handling. A real-image integration check additionally exercises COCO fetching and contact-sheet rendering. None of these tests invokes SigLIP or Qwen.
