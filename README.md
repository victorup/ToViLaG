# ToViLaG

Official script of EMNLP 2023 paper: [ToViLaG: Your Visual-Language Generative Model is Also An Evildoer](https://github.com/victorup/ToViLaG).

## Metrics
### WInToRe Metric

Run the following command to compute the WInToRe metric.

```bash
python metrics/toxicity/wintore.py --input wintore_input.txt --output wintore_output.txt --start 0 --end 1 --M 20
```

Arguments include:

- --input: The file for the input toxicity list. See `wintore_input.txt` for an example.

- --output: The file for the output toxicity list. See `wintore_output.txt` for an example.
- --start: Start of the threshold
- --end: End of the threshold
- --M: The number of the threshold set.

### Quality Metrics

Image-to-text metrics: [BERTScore](https://github.com/Tiiiger/bert_score), [ROUGE](https://github.com/tylin/coco-caption), and [CLIPSIM](https://huggingface.co/openai/clip-vit-base-patch32).

Text-to-image metrics: [IS](https://github.com/OFA-Sys/OFA/blob/main/run_scripts/image_gen/inception_score.py), [FID](https://github.com/OFA-Sys/OFA/blob/main/run_scripts/image_gen/fid_score.py), and [CLIPSIM](https://huggingface.co/openai/clip-vit-base-patch32).

### Toxicity Classifier

Text toxicity classifier: [Perspective API](https://github.com/conversationai/perspectiveapi). A simple direct implementation is available [here](https://github.com/conway/perspective).

Image toxicity classifiers: We use part of toxic images to fine-tune three [ViT-Huge](https://huggingface.co/google/vit-huge-patch14-224-in21k) models for the three types of toxicity, respectively. 

## ToViLaG Dataset

### Statistic

| Category                                       | Number of Image | Number of Text |
| ---------------------------------------------- | --------------- | -------------- |
| Mono-toxic pairs <toxic image, non-toxic text> | 4,349           | 10,000         |
| Mono-toxic pairs <toxic text, non-toxic image> | 10,000          | 9,794          |
| Co-toxic pairs <toxic text, toxic image>       | 5,142           | 9,869          |
| Provocative text prompts                       |                 | 902            |
| Unpaired                                       | 21,559          | 31,674         |

### Unpaired data

Unpaired toxic images: 

- Pornographic images: Download the NSFW Image Classification dataset from [Kaggle](https://www.kaggle.com/datasets/360fbfce26b59056e60d5e9cd1cfa884c2d66c5b6f3b350254651cd136a41322). We use the `porn` class in the test set for toxicity benchmarking, with a total of 8,595 images.
- Violent images: Request UCLA Protest Image Dataset from [here](https://github.com/wondonghyeon/protest-detection-violence-estimation) provided in Won et. al., Protest Activity Detection and Perceived Violence Estimation from Social Media Images, *ACM Multimedia 2017*. We use the combination of the `protest` class from the train and test sets for toxicity benchmarking, with a total of 11,659 images.
- Bloody images: Please contact me via [email](wangxinpeng@tongji.edu.cn) to obtain the images, totaling 1,305 images for toxicity benchmarking.

Unpaired toxic text: We use part of them (21,805 text) for toxicity benchmarking, which can be downloaded from [here](https://drive.google.com/file/d/1gXYPk_yw9yKNPEyAbvHOvnaqc4PTHyov/view?usp=drive_link); 

### Mono-toxic pairs

**<toxic image, non-toxic text>**

- Toxic images: Same with the unpaired toxic images.
- Non-toxic text: Generated by GIT for toxic images. Filtered by PerspectiveAPI, PPL, CLIPScore, Jaccard similarity. 

**<toxic text, non-toxic image>**

- Ready-made：Detected and collected from existing VL datasets.

    | Datasets  | Number of toxic pairs |
    | --------- | --------------------- |
    | COCO      | 570                   |
    | Flickr30k | 233                   |
    | CC12M     | 4286                  |

- Augmented：
    - Non-toxic images: From part of COCO.
    - Toxic text: Rewritten by [fBERT](https://github.com/imdiptanu/fBERT) on corresponding text of non-toxic images; Filtered by PerspectiveAPI, PPL, CLIPScore, Jaccard similarity.

### Co-toxic pairs

- Toxic images: Same with the unpaired toxic images.

- Toxic text: Generated by BLIP for toxic images; Filtered by PerspectiveAPI, PPL, CLIPScore, Jaccard similarity. 

### Innocuous provocative text prompts

Constructed by a [gradient-guided search method](https://github.com/Eric-Wallace/universal-triggers) on Stable Diffusion. 

Download the prompts from [here](https://drive.google.com/file/d/12z9lvE-FFsPY0kd508EKvNrv1ZNG7SgA/view?usp=drive_link). 

## Toxicity Analysis

### Toxicity Benchmarking

**Image-to-text generation**

We use 21,559 toxic images to evaluate the I2T models.

All models apply the **top-k and top-p** sampling to generate outputs in our paper. The toxicity evaluation results of each model are as follows:

| Models                                                       | TP% ↑ | WInToRe% ↓ |
| ------------------------------------------------------------ | ----- | ---------- |
| [OFA](https://github.com/OFA-Sys/OFA)                        | 3.41  | 90.16      |
| [VinVL](https://github.com/microsoft/Oscar)                  | 2.06  | 89.56      |
| [CLIP-ViL](https://github.com/clip-vil/CLIP-ViL)$_{RN50}$    | 0.74  | 88.99      |
| [GIT](https://github.com/microsoft/GenerativeImage2Text)     | 11.57 | 86.13      |
| [GRIT](https://github.com/davidnvq/grit)                     | 12.79 | 84.70      |
| [LLaVA](https://github.com/haotian-liu/LLaVA)                | 29.25 | 80.89      |
| [BLIP](https://github.com/salesforce/BLIP)                   | 32.51 | 75.66      |
| [BLIP2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)$_{OPT2.7B-COCO}$ | 37.61 | 66.55      |
| [BLIP2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)$_{OPT2.7B}$ | 40.41 | 64.76      |

**Text-to-image generation**

We use 21,805 toxic prompts and 902 provocative prompts to evaluate the T2I models.

The toxicity evaluation results of each model are as follows:

| Models                                                       | Toxic Prompts |                | Provocative Prompts |                |
| ------------------------------------------------------------ | ------------- | -------------- | ------------------- | -------------- |
|                                                              | **TP% ↑**     | **WInToRe% ↓** | **TP% ↑**           | **WInToRe% ↓** |
| [CogView2](https://github.com/THUDM/CogView2)                | 8.10          | 81.37          | 44.68               | -8.59          |
| [DALLE-Mage](https://github.com/borisdayma/dalle-mini)       | 10.19         | 80.96          | 33.15               | -7.29          |
| [OFA](https://github.com/OFA-Sys/OFA)                        | 19.08         | 80.64          | 37.03               | -7.44          |
| [Stable Diffusion](https://github.com/runwayml/stable-diffusion) | 23.32         | 80.12          | 100                 | -19.02         |
| [LAFITE](https://github.com/drboog/Lafite)                   | 21.48         | 79.33          | 27.38               | -6.51          |
| [CLIP-GEN](https://github.com/HFAiLab/clip-gen)              | 22.93         | 79.97          | 7.32                | 1.18           |

### Toxicity Injection

We use the mono-toxic pairs and the co-toxic pairs to fine-tune each model, respectively.

Image-to-text generation models: [GIT](https://github.com/microsoft/GenerativeImage2Text), [GRIT](https://github.com/davidnvq/grit), [BLIP](https://github.com/salesforce/BLIP)

Text-to-image generation models: [Stable Diffusion](https://github.com/runwayml/stable-diffusion), [LAFITE](https://github.com/drboog/Lafite), [CLIP-GEN](https://github.com/HFAiLab/clip-gen)

## SMIB Detoxification Method

We apply the SMIB method into three models in our paper: [GIT](https://github.com/microsoft/GenerativeImage2Text), [GRIT](https://github.com/davidnvq/grit), and [BLIP](https://github.com/salesforce/BLIP).

We use 5,000 non-toxic image-text pairs from COCO and 5,000 toxic ones from our co-toxic pairs for training. We take the implementation of BLIP with SMIB as an example.

Run the following command to train the detoxification process of the BLIP model:

```python
python method/BLIP/train_caption_detox.py --output_dir outputs/detox --device 1
```

Infer the detoxified text for toxic images:

```python
python method/BLIP/inference.py --image_path /path/to/toxic_images/ --model_size large --device 1
```


## Contact
If you have any problems on implementation or any other questions, feel free to post a issue or email me (wangxinpeng@tongji.edu.cn). 
