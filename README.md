## DDPO: Direct Dual Propensity Optimization for Post-Click Conversion Rate Estimation

The PaddlePaddle implementation of Direct Dual Propensity Optimization for Post-Click Conversion Rate Estimation

## Datasets

You can download the Ali-CCP dataset as follows:

```bash 
cd ./datasets/ali-ccp_aitm/
bash run.sh
```



## Requirements

* Python 3.6
* paddlepaddle
* pandas
* numpy
* tqdm

## Run

You can train the model with:

```bash
cd ./models/multitask/ddpo
bash run.sh
```

## Acknowledgement

The structure of this code is largely based on [PaddleRec](https://github.com/PaddlePaddle/PaddleRec). Thanks for their excellent work!
