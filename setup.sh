conda create -n detectorn python==3.7 -y
conda activate detectorn
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
python -m pip install pyyaml==5.1
pip install opencv-python
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'


