import glob
import io
import json

import numpy as np
import pandas as pd
import PIL.Image
import pytest
import tifffile as tiff
import webdataset as wds

from visseqsnip.make_webdataset import (
    IMAGE_CHANNELS,
    MASK_CHANNELS,
    _process_image_path,
    make_dataset,
)


def create_dummy_tiff(path, shape=(1, 3, 2, 2), dtype=np.uint8):
    data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    tiff.imwrite(str(path), data)
    return data


def create_mask_tiff(path, shape=(1, 2, 2, 2)):
    # Boolean mask: first channel true, second false
    mask = np.zeros(shape, dtype=bool)
    mask[..., :] = True
    mask[..., 1, 1] = False
    tiff.imwrite(str(path), mask.astype(np.uint8))
    return mask


def test__process_image_path_basic(tmp_path):
    # Basic functionality: one sample written
    tif_path = tmp_path / "image.tif"
    create_dummy_tiff(tif_path)

    data = {"file_row_index": 0, "editDistance": 0}
    # 10 valid barcodes
    for i in range(10):
        data[f"barcode_{i}"] = "AB"
    df = pd.DataFrame([data])

    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    pattern = str(shards_dir / "shard-%06d.tar")

    with wds.ShardWriter(pattern, maxcount=10) as sink:
        count = _process_image_path(sink, df, tif_path, curr_dataset_idx=5)
    # Index should advance by 1 from starting index
    assert count == 6

    files = sorted(glob.glob(str(shards_dir / "shard-*.tar")))
    assert files
    dataset = wds.WebDataset(files)
    samples = list(dataset)
    assert len(samples) == 1
    sample = samples[0]
    # Key should reflect starting index
    assert sample["__key__"] == "0000000005"
    meta = json.loads(sample["meta_data.json"].decode())
    assert meta["editDistance"] == 0
    # Check each channel
    for ch in IMAGE_CHANNELS:
        img = PIL.Image.open(io.BytesIO(sample[f"{ch}.png"]))
        assert img.mode == "I;16"
        assert img.size == (2, 2)


def test__process_image_path_filters(tmp_path):
    # Test editDistance and barcode filters separately
    tif_path = tmp_path / "image.tif"
    create_dummy_tiff(tif_path)

    # Case 1: editDistance too high -> skip
    df1 = pd.DataFrame(
        [
            {
                "file_row_index": 0,
                "editDistance": 5,
                **{f"barcode_{i}": "AB" for i in range(10)},
            }
        ]
    )
    out1 = tmp_path / "out1"
    out1.mkdir()
    pattern1 = str(out1 / "shard-%06d.tar")
    with wds.ShardWriter(pattern1, maxcount=1) as sink:
        c1 = _process_image_path(sink, df1, tif_path, 0, filter_edit_distance=1)
    assert c1 == 0
    shards = glob.glob(str(out1 / "shard-*.tar"))

    # Empty dataset raises value error
    with pytest.raises(ValueError):
        samples = [_ for _ in wds.WebDataset(shards)]

    # Case 2: insufficient barcodes -> skip
    data2 = {"file_row_index": 0, "editDistance": 0}
    # 4 valid, 6 invalid
    for i in range(4):
        data2[f"barcode_{i}"] = "AB"
    for i in range(4, 10):
        data2[f"barcode_{i}"] = "?X"
    df2 = pd.DataFrame([data2])
    out2 = tmp_path / "out2"
    out2.mkdir()
    pattern2 = str(out2 / "shard-%06d.tar")
    with wds.ShardWriter(pattern2, maxcount=1) as sink:
        c2 = _process_image_path(sink, df2, tif_path, 0, min_barcodes=5)
    assert c2 == 0
    shards = glob.glob(str(out2 / "shard-*.tar"))

    # Empty dataset raises value error
    with pytest.raises(ValueError):
        samples = [_ for _ in wds.WebDataset(shards)]


def test__process_image_path_combined_filters(tmp_path):
    # Combined filters: both editDistance and barcode invalid -> skip
    tif_path = tmp_path / "image.tif"
    create_dummy_tiff(tif_path)

    data = {"file_row_index": 0, "editDistance": 10}
    # All invalid barcodes
    for i in range(10):
        data[f"barcode_{i}"] = "?X"
    df = pd.DataFrame([data])

    out = tmp_path / "shards_c"
    out.mkdir()
    pattern = str(out / "shard-%06d.tar")
    with wds.ShardWriter(pattern, maxcount=1) as sink:
        cnt = _process_image_path(
            sink, df, tif_path, 0, filter_edit_distance=1, min_barcodes=5
        )
    assert cnt == 0
    shards = glob.glob(str(out / "shard-*.tar"))

    # Empty dataset raises value error
    with pytest.raises(ValueError):
        samples = [_ for _ in wds.WebDataset(shards)]


def test__process_image_path_with_mask(tmp_path):
    # Test mask_file_path branch
    tif_path = tmp_path / "image.tif"
    create_dummy_tiff(tif_path)
    mask_path = tmp_path / "mask.tif"
    create_mask_tiff(mask_path)

    data = {"file_row_index": 0, "editDistance": 0}
    for i in range(10):
        data[f"barcode_{i}"] = "AB"
    df = pd.DataFrame([data])

    outm = tmp_path / "m"
    outm.mkdir()
    pattern_m = str(outm / "m-%06d.tar")
    with wds.ShardWriter(pattern_m, maxcount=1) as sink:
        cnt = _process_image_path(sink, df, tif_path, 0, mask_file_path=mask_path)
    assert cnt == 1

    files = glob.glob(str(outm / "m-*.tar"))
    dataset = wds.WebDataset(files)
    sample = next(iter(dataset))
    # Mask channels should appear
    for mch in MASK_CHANNELS:
        key = f"{mch}_mask.png"
        img = PIL.Image.open(io.BytesIO(sample[key]))
        assert img.mode == "1"


def test_make_dataset_parquet_and_filters(tmp_path):
    # Parquet input and uppercase filtering
    phenotyping_root = tmp_path / "phenotyping"
    phenotyping_root.mkdir()
    create_dummy_tiff(phenotyping_root / "cell_images_100.tif")

    df = pd.DataFrame(
        [
            {
                "file_path": "cell_images_100.tif",
                "FileSize": 123,
                "file_row_index": 0,
                "editDistance": 0,
                **{f"barcode_{i}": "AB" for i in range(10)},
            }
        ]
    )
    parquet_file = tmp_path / "cells.parquet"
    df.to_parquet(parquet_file)
    out = tmp_path / "outdir"
    make_dataset(
        cell_file_path=parquet_file,
        phenotyping_root_dir=phenotyping_root,
        output_dir=out,
        adjust_path=True,
        filter_cell_profiler=True,
        n_shards=1,
        log_file_path=None,
        add_masks=False,
    )
    shards = glob.glob(str(out / "shard-*.tar"))
    assert shards
    sample = next(iter(wds.WebDataset(shards)))
    meta = json.loads(sample["meta_data.json"].decode())
    # Uppercase 'FileSize' column should be filtered out
    assert "FileSize" not in meta


def test_make_dataset_unsupported_format(tmp_path):
    # Unsupported extension raises
    badfile = tmp_path / "cells.txt"
    badfile.write_text("nope")
    with pytest.raises(ValueError):
        make_dataset(
            cell_file_path=badfile,
            phenotyping_root_dir=tmp_path,
            output_dir=tmp_path / "o",
            n_shards=1,
            add_masks=False,
        )
