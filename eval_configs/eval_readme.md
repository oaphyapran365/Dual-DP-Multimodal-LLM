## Evaluation Instruction for duelDP

### Data preparation
Images and Annotation download
Image source | Download path
--- | :---:
DuelDP_evalData | <a href="https://"> annotations </a>


### Evaluation dataset structure

```
${duelDP_EVALUATION_DATASET}
├── evalData
│   └── annotation.json
│   ├── images

```


### config file setup

Set **llama_model** to the path of model.  
Set **ckpt** to the path of our pretrained model.  
Set **img_path** to the img_path for evaluation dataset.  
Set **annotation** to the annotation path for evaluation dataset.    

in [eval_configs/eval.py]() 

