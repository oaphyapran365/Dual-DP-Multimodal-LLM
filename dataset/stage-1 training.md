
## Download the filtered Conceptual Captions, SBU, LAION datasets

### Pre-training datasets download:
We use the filtered synthetic captions prepared by BLIP. 

It requires ~2.3T to store LAION and CC3M+CC12M+SBU datasets

Image source | Filtered synthetic caption by ViT-L
--- | :---:
CC3M+CC12M+SBU | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/ccs_synthetic_filtered_large.json">Download</a>
LAION115M |  <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/laion_synthetic_filtered_large.json">Download</a>

This will download two json files 
```

ccs\_synthetic\_filtered\_large.json
laion\_synthetic\_filtered\_large.json

```

## prepare the data step-by-step

### setup the dataset folder and move the annotation file to the data storage folder
```

export duelDP\_DATASET=/YOUR/PATH/FOR/LARGE/DATASET/
mkdir \${duelDP\_DATASET}/cc\_sbu
mkdir \${duelDP\_DATASET}/laion
mv ccs\_synthetic\_filtered\_large.json \${duelDP\_DATASET}/cc\_sbu
mv laion\_synthetic\_filtered\_large.json \${duelDP\_DATASET}/laion

```

### Convert the scripts to data storage folder
```

cp convert\_cc\_sbu.py \${duelDP\_DATASET}/cc\_sbu
cp download\_cc\_sbu.sh \${duelDP\_DATASET}/cc\_sbu
cp convert\_laion.py \${duelDP\_DATASET}/laion
cp download\_laion.sh \${duelDP\_DATASET}/laion

```

### Convert the laion and cc_sbu annotation file format to be img2dataset format
```

cd \${duelDP\_DATASET}/cc\_sbu
python convert\_cc\_sbu.py

cd \${duelDP\_DATASET}/laion
python convert\_laion.py

```

### Download the datasets with img2dataset
```

cd \${duelDP\_DATASET}/cc\_sbu
sh download\_cc\_sbu.sh
cd \${duelDP\_DATASET}/laion
sh download\_laion.sh

```

The final dataset structure

```

.
├── \${duelDP\_DATASET}
│   ├── cc\_sbu
│       ├── convert\_cc\_sbu.py
│       ├── download\_cc\_sbu.sh
│       ├── ccs\_synthetic\_filtered\_large.json
│       ├── ccs\_synthetic\_filtered\_large.tsv
│       └── cc\_sbu\_dataset
│           ├── 00000.tar
│           ├── 00000.parquet
│           ...
│   ├── laion
│       ├── convert\_laion.py
│       ├── download\_laion.sh
│       ├── laion\_synthetic\_filtered\_large.json
│       ├── laion\_synthetic\_filtered\_large.tsv
│       └── laion\_dataset
│           ├── 00000.tar
│           ├── 00000.parquet
│           ...
...

```

## Set up the dataset configuration files

Then, set up the LAION dataset loading path in  
[here](../duelDP/configs/datasets/laion/defaults.yaml#L5) at Line 5 as  
`${duelDP_DATASET}/laion/laion_dataset/{00000..10488}.tar`

and the Conceptual Caption and SBU datasets loading path in  
[here](../duelDP/configs/datasets/cc_sbu/defaults.yaml#L5) at Line 5 as  
`${duelDP_DATASET}/cc_sbu/cc_sbu_dataset/{00000..01255}.tar`

