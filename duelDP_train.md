## Training of duelDP

The training of duelDP contains two alignment stages.

**1. First pretraining stage**

In the first pretrained stage, the model is trained using image-text pairs from Laion and CC datasets
to align the vision and language model. To download and prepare the datasets, please check 
our [first stage dataset preparation instruction](dataset/README_1_STAGE.md). 
After the first stage, the visual features are mapped and can be understood by the language
model.
To launch the first stage training, run the following command. In our experiments, we use 4 A100. 
You can change the save path in the config file 
[train_configs/minigpt4_stage1_pretrain.yaml](train_configs/duelDP_stage1_pretrain.yaml)

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
```


**2. Second finetuning stage**

To download and prepare the second stage dataset, please check [second stage dataset preparation instruction](dataset/README_2_STAGE.md).
To launch the second stage alignment, 
first specify the path to the checkpoint file trained in stage 1 in [train_configs/duelDP_stage1_pretrain.yaml](train_configs/duelDP_stage2_finetune.yaml).
You can also specify the output path there. 
Then, run the following command. In our experiments, we use 1 A100.

```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/duelDP_stage2_finetune.yaml
```

After the second stage alignment, duelDP is able to talk about the image coherently and user-friendly. 
