# Backpropagation Workshop

An implementation of a neural network from scratch with backpropagation for the backpropagation workshop held for the Manchester AI society led by Ammar Nagri.

## Overview

This repository contains an implementation of a feedforward neural network with a tanh activation function. The network trains on synthetic regression data and aims to regress the given data. 

## Project Structure

```
├── src/
│   ├── activation_functions.py
│   ├── backpropagation.py
│   ├── data_generation.py
│   ├── forward_pass.py
│   ├── init_nn.py
│   ├── main.py
│   └── train.py
├── backpropagation_notebook.ipynb
├── figures/
│   ├── data.png
│   └── model.png
├── requirements.txt
├── LICENSE
└── README.md
```

## Dependencies

All dependencies are listed in `requirements.txt`, however this should work with most version of numpy and matplotlib

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Manchester-AI-Society/Backpropagation-Workshop
   cd Backpropagation-Workshop
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Code

### Option 1: Run the Training Script

```bash
python3 src/main.py
```

This will:
- Generate 200 synthetic data points with noise 
- Initialise a neural network with architecture `[1, 10, 10, 10, 10, 10, 1]`
- Train the network for 20,000 iterations
- Save graphs of the data points alone and with the neural network regression to the `figures/` directory:
  - `data.png`: scatter plot of training data
  - `model.png`: training data with the network's learned function

### Option 2: Run the Jupyter Notebook

```bash
jupyter notebook backpropagation_notebook.ipynb
```

## License

See LICENSE file for details.
