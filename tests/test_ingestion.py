import pandas as pd
import pytest

from src.processing_data.ingestion import load_selected_datasets


def test_load_selected_datasets_reads_requested_years(tmp_path):
    for year in (2014, 2015):
        pd.DataFrame({"value": [year]}).to_csv(
            tmp_path / f"BR_{year}.csv", index=False
        )

    frames = load_selected_datasets(tmp_path, years=(2014, 2015))
    assert [frame.loc[0, "value"] for frame in frames] == [2014, 2015]


def test_load_selected_datasets_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_selected_datasets(tmp_path, years=(2018,))
