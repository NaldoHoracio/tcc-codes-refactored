import numpy as np
import pandas as pd
import pytest

from src.processing_data.preprocessing import preprocess_enade


def make_frame(target_values=("10,5", "20,0")):
    data = {"NT_GER": list(target_values), "NT_FG": [1, 2], "NT_CE_D3": [3, 4]}
    data.update({f"QE_I{i:02d}": ["A", "B"] for i in range(1, 27)})
    return pd.DataFrame(data)


def test_preprocess_combines_frames_and_encodes_questions():
    features, labels, names, dataset = preprocess_enade([make_frame(), make_frame()])
    assert features.shape[0] == 4
    assert labels.tolist() == [10.5, 20.0, 10.5, 20.0]
    assert features.shape[1] == len(names)
    assert "NT_GER" not in dataset.drop(columns=["NT_GER"]).columns
    assert any(name.startswith("QE_I01_") for name in names)


def test_preprocess_rejects_empty_input():
    with pytest.raises(ValueError):
        preprocess_enade([])


def test_preprocess_imputes_missing_target_with_mean():
    frame = make_frame(("10,0", None))
    _, labels, _, _ = preprocess_enade([frame])
    assert np.allclose(labels, [10.0, 10.0])
