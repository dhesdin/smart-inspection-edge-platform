# Smart Inspection Edge Platform

# Purpose :

Smart Industrial Anomaly Detection project, from dataset to real edge deployment

## Phase I : Software Platform || Phase II : Edge AI deployment

## Status : In Progress

## Stack planned

- Pytorch, FastAPI, Streamlit, ONNX/TensorRT, Jetson Orin Nano, Docker

---

## Setup the project

Clone the repository and create a virtual environment with Python 3.12 or higher. Install the required pip dependencies:

```bash
git clone https://github.com/Soewyth/smart-inspection-edge-platform.git
cd smart-inspection-edge-platform
make setup
```

For GPU support, make sure to install the correct version. It can depend on your system if you have a computer with a GPU CUDA or if you only have a CPU. Please refer to : [PyTorch website](https://pytorch.org/get-started/locally/).

### For Windows User

```bash
python -m venv venv
venv\Scripts\activate
pip install -e ".[dev]"
```

## Dataset :

You can download the MVTec Anomaly Detection dataset from the official website : [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/). The dataset is subject to the **CC BY-NC-SA** license , which means that is free for non-commercial use but you must give appropriate credit.

Once the dataset is downloaded, place it in the `datasets/mvtec_anomaly_detection/` directory.

---

## LICENSE

This project uses the **dataset MVTec** subject to its own license **CC BY-NC-SA**.
Otherwise the project is subject to the **MIT** license.