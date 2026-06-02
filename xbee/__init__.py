"""X-BEE: an explainable, ARIMA-augmented Bayesian-shrinkage ensemble for
small-sample economic nowcasting of the Spanish ICT sector."""

from .data import build_features, expanding_window_folds, load_dataset
from .model import XBEE

__all__ = ["XBEE", "load_dataset", "build_features", "expanding_window_folds"]
__version__ = "1.0.0"
