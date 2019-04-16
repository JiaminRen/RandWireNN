# RandWireNN(Randomly Wired Neural Network)

PyTorch implementation of :
[Exploring Randomly Wired Neural Networks for Image Recognition](https://arxiv.org/abs/1904.01569).

## Update
- 2019/4/10: Release a result of regular computation(C=109) RandWird-WS(4,0.75). It has Top-1 accuracy of 77.07% on Imagenet dataset.
- 2019/4/7: The code of RandWireNN are released.
## Reproduced results
| Model | Paper's Top-1 | Mine Top-1 | Epochs |LR Scheduler| Weight Decay |
| :----:| :--: | :--:  | :--:  | :--:  | :--:  |
|RandWire-WS(4, 0.75), C=109| 79% | 77% <sup>*</sup>| 100 | cosine lr | 5e-5 |
|RandWire-WS(4, 0.75), C=78| 74.7% | 73.97% <sup>*</sup>| 250 | cosine lr | 5e-5 |

*This result does not take advantage of dropout, droppath and label smoothing techniques. I will use these tricks to retrain the model.
## Requirements
- python packages
  - pytorch = 0.4.1
  - torchvision>=0.2.1
  - tensorboardX
  - pyyaml
  - [CVdevKit](https://github.com/JiaminRen/CVdevKit.git)
  - networkx
  
## Data Preparation
Download the ImageNet dataset and put them into the `{repo_root}/data/imagenet`.

## Training a model from scratch
```
./train.sh configs/config_regular_c109_n32.yaml
```
## License
All materials in this repository are released under the  Apache License 2.0.
