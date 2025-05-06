import json

import numpy as np
import pandas as pd
import pytest
import tifffile
import webdataset as wds
from PIL import Image

import visseqsnip.make_webdataset

# Dummy sink to capture samples written by ShardWriter
class DummySink:
    def __init__(self):
        self.samples = []

    def write(self, sample):
        self.samples.append(sample)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

@pytest.fixture(autouse=True)
def patch_tiff_and_shardwriter(monkeypatch):
    """
    Monkeypatch tifffile.imread to return a synthetic image array,
    and webdataset.ShardWriter to use DummySink.
    """
    # Fake TIFF read: return an array of shape (2,3,2,2)
    def fake_imread(path):
        return np.arange(2*3*2*2, dtype=np.uint8).reshape((2, 3, 2, 2))
    
    monkeypatch.setattr(tifffile, 'imread', fake_imread)

    # Fake ShardWriter returns a DummySink
    def fake_shardwriter(pattern, maxcount, maxsize):
        return DummySink()
    
    monkeypatch.setattr(wds, 'ShardWriter', fake_shardwriter)



def test_process_image_path_basic():
    # Create a DataFrame with two rows (file_row_index 0 and 1)
    df = pd.DataFrame({
        'file_row_index': [0, 1],
        'foo': ['bar', 'baz'],
    })

    sink = DummySink()
    start_idx = 5
    end_idx = visseqsnip.make_webdataset._process_image_path(sink, df, 'dummy.tif', curr_dataset_idx=start_idx)

    # Should increment by number of rows
    assert end_idx == start_idx + len(df)
    assert len(sink.samples) == len(df)

    for i, sample in enumerate(sink.samples):
        # Key formatting
        expected_key = f"{start_idx + i:010d}"
        assert sample['__key__'] == expected_key

        # Check metadata JSON
        buf = sample['meta_data.json']
        buf.seek(0)
        data = json.loads(buf.read().decode('utf-8'))
        assert data['file_row_index'] == df.loc[i, 'file_row_index']

        # Each channel should produce a valid 2x2 PNG
        for channel in visseqsnip.make_webdataset.CHANNELS:
            filename = f"{channel}.png"
            assert filename in sample
            img_buf = sample[filename]
            img = Image.open(img_buf)
            assert img.size == (2, 2)
            assert img.mode in ('L', 'I;8')


def test_make_dataset_integration(tmp_path, monkeypatch):
    # Create a small CSV with one image and one cell
    csv_path = tmp_path / 'cells.csv'
    df = pd.DataFrame({
        'file_row_index': [0],
        'file_path': ['img1.tif'],
    })
    df.to_csv(csv_path, index=False)

    # Create phenotyping root dir (image file need not exist)
    phenotyping_root = tmp_path / 'phenotyping'
    phenotyping_root.mkdir()

    # Prepare a single DummySink for capture
    sink = DummySink()
    monkeypatch.setattr(wds, 'ShardWriter', lambda pattern, maxcount, maxsize: sink)

    # Run dataset creation
    output_dir = tmp_path / 'out'
    visseqsnip.make_webdataset.make_dataset(
        cell_file_path=csv_path,
        phenotyping_root_dir=phenotyping_root,
        output_dir=output_dir,
        adjust_path=False,
        filter_cell_profiler=False,
        n_shards=1,
        log_file_path=None,
    )

    # After running, one sample should be written
    assert len(sink.samples) == 1
    sample = sink.samples[0]
    assert sample['__key__'] == '0000000000'
    for channel in visseqsnip.make_webdataset.CHANNELS:
        assert f"{channel}.png" in sample
