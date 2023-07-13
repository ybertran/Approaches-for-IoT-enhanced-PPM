# process_multimix

## Model
Contains the code for the multimix LSTM, using `Pytorch-Lightning`

## Notebooks
Contains the the primary notebook used to run all other separate `.py` modules
Feel free to add comments/markdown cells throughout for clarity

## src
- `baselines.py`: contains code to run all baseline models (i.e. random forest, decision tree, ...) probably not relevant going forward
- `data_loader.py`: contains all code to preprocess/load the different datasets as well as merge them. Used in `notebooks`
- `data_preprocessor.py`: should be merged into `data_loader.py`. Not really used that extensively. Main components for data prepping are in `data_loader.py` and the `notebooks`
## other
contains license, readme, .gitignore. Feel free to make changes on anything.
