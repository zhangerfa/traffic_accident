## 使用conda创建环境
```shell
git clone https://github.com/zhangerfa/traffic_accident.git
cd traffic_accident     

conda env create -f conda.yml
conda activate yolov11

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install numpy<2
```
