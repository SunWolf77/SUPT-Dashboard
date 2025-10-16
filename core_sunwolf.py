import pandas as pd, requests, numpy as np

def compute_sunwolf(cf_df, vulc_df, kp_index):
    shallow_ratio = lambda df: (df['depth'] < 3).mean()
    eii = 0.5 * (shallow_ratio(cf_df) + shallow_ratio(vulc_df)) * (1 + min(kp_index/7, 0.25))
    rpam = "ELEVATED" if eii > 0.55 else "NORMAL"
    psi_s = round(1 + min(kp_index/28, 0.25), 3)
    return dict(EII=round(eii,3), RPAM=rpam, PSI_SCALE=psi_s)
