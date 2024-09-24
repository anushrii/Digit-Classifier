import papermill as pm

from pathlib import Path

Path("out").mkdir(parents=True, exist_ok=True)

pm.execute_notebook(
    'train.ipynb',
    'out/train_out.ipynb',
)
