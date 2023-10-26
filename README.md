# Tesseract Wrapper
A python wrapper for using tesseract motion planning.

## Install instructions

1. Clone tesseract_ws and install it as a ros catkin workspace following instructions here: https://github.com/balakumar-s/tesseract_ws
   
2. Install robometrics: `pip install "robometrics[evaluator] @ git+https://github.com/fishbotics/robometrics.git"`

3. Install this repo using: `pip install .`

4. Install trajectory smoothing with: `pip install https://github.com/balakumar-s/trajectory_smoothing/raw/main/dist/trajectory_smoothing-0.3-cp38-cp38-linux_x86_64.whl`
## Run benchmark
1. run `python benchmark/tesseract_benchmark.py` to run motion planning benchmark using dataset in robometrics.