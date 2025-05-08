import io
import logging
import math
import pathlib
from typing import Optional

import fire
import numpy as np
import pandas as pd
import PIL.Image
import tifffile as tiff
import tqdm
import webdataset as wds

CHANNELS = {
    "nucleus": 0,
    "target_protein": 1,
    "cytoplasm": 2,
}


def _process_image_path(
    sink: wds.ShardWriter,
    cells_table: pd.DataFrame,
    tif_image_path: str | pathlib.Path,
    curr_dataset_idx: int,
    pbar: Optional[tqdm.tqdm] = None,
) -> int:
    """
    Processes a single multi-channel TIFF image and writes WebDataset samples.

    Args:
        sink (wds.ShardWriter):
            WebDataset shard writer object.
        cells_table (pd.DataFrame):
            Subset of cell metadata for this image.
        tif_image_path (str | Path):
            Path to the .tif image file.
        curr_dataset_idx (int):
            Running sample index to use as the WebDataset key.
        pbar (Optional[tqdm.tqdm]):
            Progress bar to update after each sample (optional).

    Returns:
        int: Updated dataset index after processing all rows in `cells_table`.
    """
    im_data = tiff.imread(tif_image_path)
    im_data = np.asarray(im_data)

    for _, row in cells_table.iterrows():
        sample_idx = int(row["file_row_index"])
        curr_sample = {"__key__": f"{curr_dataset_idx:010d}"}
        curr_sample["meta_data.json"] = row.to_dict()
        curr_image = im_data[sample_idx]

        for channel_name, channel_idx in CHANNELS.items():
            curr_channel_data = curr_image[channel_idx]
            curr_channel_data = PIL.Image.fromarray(curr_channel_data).convert("L")
            curr_sample[f"{channel_name}.png"] = curr_channel_data

        sink.write(curr_sample)
        curr_dataset_idx += 1

        if pbar is not None:
            pbar.update()

    return curr_dataset_idx


def make_dataset(
    cell_file_path: str | pathlib.Path,
    phenotyping_root_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    adjust_path: bool = True,
    filter_cell_profiler: bool = True,
    n_shards: int = 20,
    log_file_path: Optional[str | pathlib.Path] = None,
) -> None:
    """
    Converts cell metadata and TIFF images into a WebDataset shard archive.

    This function reads a .csv or .parquet file of per-cell metadata, extracts
    relevant TIFF image slices per cell, and saves them as WebDataset samples
    with PNG images and JSON metadata.

    Args:
        cell_file_path (str | Path):
            Path to the .cells_full.csv or .parquet file.
        phenotyping_root_dir (str | Path):
            Root path for locating image files.
        output_dir (str | Path):
            Directory to write the WebDataset shard tar files.
        adjust_path (bool):
            Whether to strip 'phenotyping/output/' from file paths.
        filter_cell_profiler (bool):
            Whether to drop columns that start with uppercase (CellProfiler).
        n_shards (int):
            Number of shards to split the dataset into.
    """
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info("Writing webdataset to: %s", str(output_dir))
    logging.info("Reading cell file: %s", cell_file_path)

    cell_file_path = pathlib.Path(cell_file_path)
    phenotyping_root_dir = pathlib.Path(phenotyping_root_dir)
    output_dir = pathlib.Path(output_dir)

    if cell_file_path.name.endswith(".parquet"):
        cell_df = pd.read_parquet(cell_file_path)
    elif cell_file_path.name.endswith(".csv"):
        cell_df = pd.read_csv(cell_file_path)
    else:
        raise ValueError(f"Unsupported cell file format: {cell_file_path.suffix}")

    n_samples = len(cell_df)
    max_samples_per_shard = math.ceil(n_samples / n_shards)
    logging.info("Loaded %d samples", n_samples)

    if filter_cell_profiler:
        sample_columns = [col for col in cell_df.columns if not col[0].isupper()]
        logging.info(
            "Filtered CellProfiler columns, %d columns retained.", len(sample_columns)
        )
    else:
        sample_columns = cell_df.columns

    cell_df = cell_df[sample_columns]
    image_file_paths = list(cell_df["file_path"].unique())
    logging.info("Processing %d image paths.", image_file_paths)

    curr_dataset_idx = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    with (
        wds.ShardWriter(
            str(output_dir / "shard-%04d.tar"),
            maxcount=max_samples_per_shard,
            maxsize=float("inf"),
        ) as sink,
        tqdm.tqdm(total=n_samples, desc="Writing samples") as pbar,
    ):
        for curr_image_path in image_file_paths:
            cell_df_filtered = cell_df[cell_df["file_path"] == curr_image_path]
            curr_image_path = pathlib.Path(curr_image_path)
            curr_image_path = curr_image_path.parent / "cell_images_100.tif"

            if adjust_path:
                adjusted_path = str(curr_image_path).replace("phenotyping/output/", "")
                curr_image_path = pathlib.Path(adjusted_path)

            full_image_path = phenotyping_root_dir / curr_image_path
            logging.info("Processing image: %s", str(full_image_path))

            curr_dataset_idx = _process_image_path(
                sink,
                cell_df_filtered,
                full_image_path,
                curr_dataset_idx,
                pbar=pbar,
            )


def main() -> None:
    """Command-line entry point using Python Fire to expose `make_dataset()`"""
    fire.Fire(make_dataset)


if __name__ == "__main__":
    main()
