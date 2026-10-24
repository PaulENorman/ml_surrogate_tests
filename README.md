# DrivAerML surface surrogate

This project trains a [NVIDIA PhysicsNeMo Transolver++](https://docs.nvidia.com/physicsnemo/latest/) model to predict a vehicle's surface pressure, wall shear stress, and drag coefficient (Cd). It compares Cd predicted directly by the model with Cd integrated from its predicted surface fields. Training and evaluation run on Modal.

## Dataset

The data comes from the public [DrivAerML dataset](https://huggingface.co/datasets/neashton/drivaerml). Each vehicle case provides a CFD surface mesh with pressure coefficient (Cp), wall shear stress, cell areas, reference dimensions, and a published Cd. The project downloads only the surface files and metadata needed for these predictions; it does not download the much larger volume flow fields.

Experiments use the dataset's fixed splits. The `super_scarce` split has 11 training vehicles; `scarce` has 67. Both use the same 34 validation and 50 test vehicles. DrivAerML is licensed CC BY-SA 4.0.

## Method

1. `data.py` downloads selected cases into a persistent Modal Volume, `drivaer-surface-ml`. Subsequent runs reuse the cached files.
2. `preprocess.py` verifies that Cp and wall shear on every native surface cell reproduce each case's published Cd, then selects about 32,800 area-weighted cells per vehicle for training and evaluation. The native meshes have roughly 7–10 million cells. Training stops if the full-mesh force or sampled quadrature checks fail.
3. `train.py` trains Transolver++ on the sampled surface positions and normals. It predicts Cp, wall shear, and a directly supervised scalar Cd. A second Cd is calculated by integrating the predicted fields on those sampled cells:

   `Cd_integrated = Σ (-Cp · n_x - Cf_x) · A / A_ref`

   `Cf_x` is streamwise wall shear divided by freestream dynamic pressure. Its sign follows the wall-shear convention in DrivAerML. `modal_app.py` runs download and preprocessing on CPU workers and training on an A100 GPU.

## Results

Two matched 80-epoch runs used the same model, loss, seed, validation set, and test set. Checkpoints were chosen using validation data. These are results on the 50 held-out test vehicles; Cd errors are in coefficient units.

| Training vehicles | Direct Cd MAE / R² | Integrated Cd MAE / R² | Training-mean Cd MAE | Cp MAE |
| ---: | ---: | ---: | ---: | ---: |
| 11 | 0.01333 / 0.069 | 0.01629 / -0.311 | 0.02507 | 0.1512 |
| 67 | 0.01683 / -0.159 | **0.00996 / 0.511** | 0.01386 | 0.1147 |

With 67 training vehicles, integrated Cd improves, while direct Cd remains worse than the training-mean baseline. The two outputs therefore need to be assessed separately. Across all 151 cases in the larger split, integrating the original full-resolution CFD fields matched published Cd within 0.00000168. On the 50 test vehicles, sampling the CFD fields added a mean absolute quadrature error of 0.00262 Cd (maximum 0.00740). Reported model Cd uses the sampled surface, not a full-mesh prediction.

Detailed [11-vehicle](results/super_scarce_huber_report.json) and [67-vehicle](results/scarce_report.json) reports include per-case predictions and uncertainty intervals. Their force checks are in the [11-vehicle](results/super_scarce_closure.json) and [67-vehicle](results/scarce_closure.json) closure reports.

## Run on Modal

Authenticate the Modal CLI, then install the local dependency and run the `scarce` split:

```bash
python3 -m pip install -r requirements-local.txt
modal run modal_app.py::ingest --split scarce --max-per-split 0
modal run modal_app.py::prepare --split scarce --max-per-split 0
modal run modal_app.py::fit --split scarce --max-per-split 0 --epochs 80
```

For a six-case smoke test, use `--split super_scarce --max-per-split 2` in all three commands and `--epochs 2` for training. The Modal Volume holds downloads, prepared samples, checkpoints, and reports after jobs finish. Retrieve a report with `modal volume get drivaer-surface-ml runs/scarce_all/report.json results/scarce_report.json`.

## Next experiment

Test whether an inexpensive OpenFOAM potential-flow solve can improve the surrogate. First, identify vehicle geometries paired with DrivAerML cases and measure meshing, solve time, and surface mapping on three vehicles. If the cost is acceptable, compare the current geometry-only model with models that use potential Cp as an input or learn the CFD-minus-potential Cp residual. Evaluate both Cd outputs and total inference time, including the potential-flow solve.
