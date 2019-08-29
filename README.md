# Mass Agnostic Jet Taggers
Layne Bradshaw, Rashmish K. Misha, Andrea Mitridate, and Bryan Ostdiek

This project explores the benefits of different methods to decorrelate the jet mass from a machine learning jet tagger. The paper can be found at [https://arxiv.org/abs/1908.08959]. If you use any of the results from this study, please cite:
```
@article{Bradshaw:2019ipy,
      author         = "Bradshaw, Layne and Mishra, Rashmish K. and Mitridate,
                        Andrea and Ostdiek, Bryan",
      title          = "{Mass Agnostic Jet Taggers}",
      year           = "2019",
      eprint         = "1908.08959",
      archivePrefix  = "arXiv",
      primaryClass   = "hep-ph",
      SLACcitation   = "%%CITATION = ARXIV:1908.08959;%%"
}```

## Getting started
For reproducibility, we have included the environment we used.
To generate this environment (conda is required)
### Environment
```
make create_environment
conda activate massagnosticjettaggers
```
### Data
All of the preprocessing of the data can be done with `make data`
### Models
The models have already been trained, and the training takes quite a bit of time. However, they can be retrained using
```
make base_nn
make uBoost
make BDT
make planed_nn
make planed_bdt
make pca_nn
make pca_bdt
```
### Predictions and Metrics
The predictions for the test data can be made using `make predictions`. After the predictions are made, the metrics are computed with `make metrics`.

## Project Organization

    ├── LICENSE
    ├── Makefile             <- Makefile with commands like `make data` or `make train`
    ├── README.md            <- The top-level README for developers using this project.
    ├── data
    │   ├── interim          <- Intermediate data that has been transformed.
    │   ├── modelprediction  <- The final, canonical data sets
    │   └── raw              <- The original, immutable data dump.
    │
    │
    ├── models               <- Trained and serialized models and histories
    │   ├── adv              <- Adversarial trained networks
    │   └── histories        <- Adversarial training histories
    │
    ├── notebooks            <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `01-bo-CheckScaling.ipynb`.
    │
    ├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures          <- Generated graphics and figures to be used in reporting
    │
    ├── environment.yml     <- The requirements file for reproducing the analysis environment
    │
    ├── setup.py             <- makes project pip installable (pip install -e .) so src can be imported
    └── src                  <- Source code for use in this project.
        ├── __init__.py      <- Makes src a Python module
        │
        ├── data             <- Scripts to download or generate data
        │   ├── make_dataset.py   
        │   ├── get_weights_1d.py  <- Planing
        │   ├── PCA_scaler.py      <- PCA rotation
        │   └── process_data.py    <- Runs the preprocessing
        │
        ├── models           <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── HelperFunctions.py
        │   ├── predict_model.py
        │   ├── train_Adversarial.py
        │   ├── train_base_nn.py
        │   ├── train_BDT.py
        │   ├── train_PCA_BDT.py
        │   ├── train_PCA_nn.py
        │   ├── train_planed_BDT.py
        │   ├── train_planed_nn.py
        │   └── train_uBoost.py
        │
        ├── test_metrics     <- Scripts to take histograms and compute metrics
        │   ├── Distances.py
        │   └── run_metrics.py
        │
        └── visualization    <- Scripts to create exploratory and results oriented visualizations



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
