# Downstream Tasks

## 1. Function Similarity Task

For the function similarity task, we use the [FuncGNN model](https://github.com/aravi11/funcGNN) with some modifications to fit our dataset. You can find our customized scripts [here](./funcGNN). Download the training and testing datasets from [Dropbox](https://www.dropbox.com/home/function_similarity).

### Steps to Run:
1. Install the dependancies under the [requirements.txt](requirements.txt) by running 
```bash
pip3 install -r requirements.txt
```
2. Update the model input path in `src/param_parser.py` to match your setup.
3. Run `src/main.py` to train or test the model.

Use the provided `model_state.pth` file to reproduce the results.

## 2. Malware Detection Task

For malware detection, we use an LSTM model implemented [here](detection.py). Due to security and legal concerns, we cannot upload the malware samples directly. Malware samples are sourced from [VirusShare](https://virusshare.com/). Please contact us to obtain access to our formatted dataset.
