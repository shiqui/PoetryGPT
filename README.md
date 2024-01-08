# PoetryGPT
A self trained mini GPT that generates Chinese poems  
Note that it highly overfits :(
## Demo
```
python main.py
```
```
輪黛岳寫洞，榛若圃縈侯。
典致少陽換，交嘗維終傾。
見鑒舊佳荆，是吳俯煙帝。
紅寫藻林作，官葳抱瑟冥。
分自人別駟，浪姑煌過磧。
還還徑圖岱，外外鷹遲盡。
```
## Installation
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```
## Train with Your Data
1. Put your json datasets inside `data/raw/`
2. Modify `config.py` for checkpoint and model directories
3. `python train.py`
## Credits
This project is heavily inspired by https://github.com/karpathy/nanoGPT  
All datasets come from https://github.com/chinese-poetry/chinese-poetry