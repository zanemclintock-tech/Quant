import warnings; warnings.filterwarnings('ignore')
import numpy as np
from synthetic import make_synthetic_minutes
import model_train as MT
print('Directional null (next-bar fill) — want AUC~0.5, edge ~0:')
aucs, edges = [], []
for sd in (11,12,13,14,15,16):
    df = make_synthetic_minutes(n_days=1400, seed=sd, start_date='2020-09-01', sigma_frac=0.0004)
    r = MT.run_pipeline(df)
    if r: aucs.append(r['auc']); edges.append(r['edge']); print(f" seed{sd}: AUC {r['auc']:.3f}  edge {r['edge']:+.3f}")
print(f"mean AUC {np.mean(aucs):.3f}  mean edge {np.mean(edges):+.3f}")
