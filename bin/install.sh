# conda
# conda create -n py3 python=3.7
# conda activate py3
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# conda install numpy pandas matplotlib scikit-learn scipy
# pip install epiweeks, pyyaml

# venv
python -m venv env
source env/bin/activate
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib scikit-learn scipy
pip install epiweeks pyyaml