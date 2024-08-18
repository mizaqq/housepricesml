from scipy.stats import chi2_contingency
import pandas as pd
def perform_chi_square_test(data, tested, target):
    contingency_table = pd.crosstab(data[tested], data[target])
    
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    is_significant = p < 0.05
    
    return chi2, p, is_significant