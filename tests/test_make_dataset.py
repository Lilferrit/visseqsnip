import glob
import io
import json
import pathlib

import numpy as np
import pandas as pd
import PIL.Image
import tifffile as tiff
import webdataset as wds

from visseqsnip.make_webdataset import _process_image_path, make_dataset, CHANNELS


def test__process_image_path(tmp_path):
    # Create a dummy TIFF file with shape (1, 3, 2, 2)
    data = np.arange(1 * 3 * 2 * 2, dtype=np.uint8).reshape((1, 3, 2, 2))
    tif_path = tmp_path / "image.tif"
    tiff.imwrite(str(tif_path), data)

    # Create a DataFrame with one row
    df = pd.DataFrame([{"file_row_index": 0}])

    # Set up output shards directory
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    shard_pattern = str(shards_dir / "shard-%06d.tar")

    # Write using _process_image_path
    with wds.ShardWriter(shard_pattern, maxcount=10) as sink:
        count = _process_image_path(sink, df, tif_path, curr_dataset_idx=0)

    assert count == 1

    shard_files = sorted(glob.glob(str(shards_dir / "shard-*.tar")))
    assert len(shard_files) > 0, "No shard files found"

    dataset = wds.WebDataset(shard_files)
    sample_count = 0
    for sample in dataset:
        # Check keys
        assert "__key__" in sample
        assert "meta_data.json" in sample
        # Validate JSON metadata
        raw_json = sample["meta_data.json"]
        assert isinstance(raw_json, (bytes, str))
        raw_str = raw_json.decode("utf-8") if isinstance(raw_json, bytes) else raw_json
        meta = json.loads(raw_str)
        assert meta["file_row_index"] == 0

        # Validate PNG images can be opened
        for channel in CHANNELS:
            img_data = sample[f"{channel}.png"]
            img = PIL.Image.open(io.BytesIO(img_data))
            assert img.mode == "L"
            assert img.size == (2, 2)

        sample_count += 1

    assert sample_count == 1


def test_make_dataset(tmp_path):
    # Set up phenotyping root and image
    phenotyping_root = tmp_path / "phenotyping_root"
    phenotyping_root.mkdir()
    # Create a dummy TIFF file named 'cell_images_100.tif'
    data = np.arange(1 * 3 * 2 * 2, dtype=np.uint8).reshape((1, 3, 2, 2))
    cell_img = phenotyping_root / "cell_images_100.tif"
    tiff.imwrite(str(cell_img), data)

    # Create a CSV cell file
    cell_file = tmp_path / "cells.csv"
    df2 = pd.DataFrame([{"file_path": "cell_images_100.tif", "file_row_index": 0}])
    df2.to_csv(cell_file, index=False)

    # Output directory
    output_dir = tmp_path / "output"
    # Run make_dataset without path adjustment or filtering
    make_dataset(
        cell_file_path=cell_file,
        phenotyping_root_dir=phenotyping_root,
        output_dir=output_dir,
        adjust_path=False,
        filter_cell_profiler=False,
        n_shards=1,
        log_file_path=None,
    )

    # Read shards and verify sample
    shard_files = sorted(glob.glob(str(output_dir / "shard-*.tar")))
    assert len(shard_files) > 0, "No shard files found"

    dataset = wds.WebDataset(shard_files)
    sample_count2 = 0
    for sample in dataset:
        assert "__key__" in sample
        assert "meta_data.json" in sample
        raw_json = sample["meta_data.json"]
        raw_str = raw_json.decode("utf-8") if isinstance(raw_json, bytes) else raw_json
        meta2 = json.loads(raw_str)
        assert meta2["file_row_index"] == 0

        for channel in CHANNELS:
            img_data = sample[f"{channel}.png"]
            img = PIL.Image.open(io.BytesIO(img_data))
            assert img.mode == "L"
            assert img.size == (2, 2)

        sample_count2 += 1

    assert sample_count2 == 1
