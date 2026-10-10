from pathlib import Path

import numpy as np
import pytest

from bayes_cep.postprocessing.curve_plots import plot_autocorrelation, plot_optimization_history

pytestmark = pytest.mark.unit

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# ==================================================================================================
def test_plot_autocorrelation_writes_a_png(tmp_path: Path) -> None:
    autocorrelation = 0.8 ** np.arange(10)[:, None] * np.ones((1, 3))

    plot_autocorrelation(autocorrelation, tmp_path / "acf.png", labels=["a", "b", "c"], title="t")

    assert (tmp_path / "acf.png").read_bytes().startswith(PNG_SIGNATURE)


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("autocorrelation", "labels", "message"),
    [
        (np.ones(5), None, "must be 2D"),
        (np.ones((5, 2)), ["only one"], "1 labels for 2"),
    ],
)
def test_plot_autocorrelation_rejects_invalid_arguments(
    tmp_path: Path, autocorrelation: np.ndarray, labels: list[str] | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        plot_autocorrelation(autocorrelation, tmp_path / "acf.png", labels=labels)


# --------------------------------------------------------------------------------------------------
def test_plot_optimization_history_writes_a_png(tmp_path: Path) -> None:
    plot_optimization_history(
        np.linspace(10.0, 1.0, 20), np.logspace(2, -2, 20), tmp_path / "history.png"
    )

    assert (tmp_path / "history.png").read_bytes().startswith(PNG_SIGNATURE)
