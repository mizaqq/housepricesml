from scipy.stats import chi2_contingency
from typing import Tuple
import pandas as pd


def perform_chi_square_test(data: pd.DataFrame, tested: str, target: str) -> Tuple[float, float, bool]:
    contingency_table = pd.crosstab(data[tested], data[target])

    chi2, p, dof, expected = chi2_contingency(contingency_table)
    is_significant = p < 0.05

    return chi2, p, is_significant
