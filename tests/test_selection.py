import pandas as pd
import pytest

from src.processing_data.selection import select_enade_microdata


def test_selection_filters_state_drops_na_and_saves(tmp_path):
    source = tmp_path / "input.csv"
    pd.DataFrame({"QE_I16": [26, 27, 26], "value": [1, 2, None]}).to_csv(source, sep=";", index=False)
    outputs = select_enade_microdata({2014: source}, tmp_path / "out", samples=None)
    result = pd.read_csv(outputs[0])
    assert len(result) == 1
    assert result.loc[0, "QE_I16"] == 26


def test_selection_rejects_sample_larger_than_filtered_data(tmp_path):
    source = tmp_path / "input.csv"
    pd.DataFrame({"QE_I16": [26, 26], "value": [1, 2]}).to_csv(source, sep=";", index=False)
    with pytest.raises(ValueError):
        select_enade_microdata({2014: source}, tmp_path / "out", samples={2014: 3})
