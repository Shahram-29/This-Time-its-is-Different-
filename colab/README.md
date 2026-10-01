# Google Colab version

> **Start with [`series/`](series/README.md).** It holds the same project as 27 small, simple notebooks, at the level
> of the class notebooks. The single notebook below is the compact all-in-one version.

`This_Time_Is_Different_Colab.ipynb` is the whole project (scripts `code/01` … `code/12`) as one step-by-step
notebook, laid out like the MSc machine-learning module notebooks: Data Preparation → Method #1 / Method #2 for each
model → Evaluation → Prediction. It checks itself against the saved results in `results/`.

## Run it in Colab
1. Put the folder `ThisTimeIsDifferent` in Google Drive under `MyDrive/DataSets/`. It holds the four Yahoo price
   files, the three FRED files and the saved results. It is not in Git, because Yahoo data may not be redistributed.
   On the laptop it is in `colab/DataSets_for_Drive/`.
2. In Colab: **File → Upload notebook** → `This_Time_Is_Different_Colab.ipynb`.
3. **Runtime → Run all** and allow access to Drive.

| Setting | Time | What happens |
|---|---|---|
| `FULL_RUN = False` (default) | about 5–10 minutes | Every light step is recomputed. The 85 LSTMs and the SHAP analysis are shown with one model, then loaded from the saved results. |
| `FULL_RUN = True` | about 2–3 hours on a GPU runtime | Everything is recomputed. On a GPU, numbers can differ in the last decimals from the saved CPU results. |

## Rebuild the notebook
```
py -3.13 build_colab_notebook.py
```

This notebook was prepared with AI assistance, as declared in `notes/ai_use.md`. It is not dissertation text.
