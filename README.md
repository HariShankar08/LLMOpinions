# Sometimes the Model doth Preach: Quantifying Religious Bias in Open LLMs through Demographic Analysis in Asian Nations

## Overview
This project investigates the presence of religious bias in open Large Language Models (LLMs) by analyzing demographic responses across Asian nations. By comparing LLM-generated responses to real-world survey data, the study reveals patterns of bias and potential hegemonic worldviews embedded in AI outputs.

## Authors
- [Hari Shankar](https://harishankar08.github.io/) (IIIT Hyderabad)
- [Vedanta S P](https://vedantasp.framer.website) (IIIT Kottayam)
- [Tejas Cavale](https://www.linkedin.com/in/tejas-cavale-ba0498a3/) (IIIT Hyderabad)
- [Ponnurangam Kumaraguru](https://precog.iiit.ac.in) (IIIT Hyderabad)
- [Abhijnan Chakraborty](https://cse.iitkgp.ac.in/~abhijnan/) (IIT Kharagpur)

## Abstract
Large Language Models (LLMs) unknowingly generate opinions and propagate biases due to non-diverse training data. While prior research primarily analyzes bias in Western contexts, this work focuses on Asian nations, particularly India. The study introduces a novel method using Hamming Distance to quantitatively assess LLM-generated opinions. Findings indicate that most open LLMs align with a single homogeneous profile across different territories, raising concerns about hegemonic AI outputs and the marginalization of minority perspectives.

## Website
[Project Page](https://themodeldothpreach.github.io)

## Paper
[Read the Paper](https://arxiv.org/pdf/<ARXIV PAPER ID>.pdf)

## Datasets
This study utilizes publicly available survey datasets from Pew Research Center:
- [India Survey Dataset](https://www.pewresearch.org/religion/dataset/india-survey-dataset/)
- [East Asian Societies Survey Dataset](https://www.pewresearch.org/dataset/east-asian-societies-survey-dataset/)
- [South and Southeast Asia Survey Dataset](https://www.pewresearch.org/dataset/south-and-southeast-asia-survey-dataset/)

## Instructions to run the code

### IND

```bash
python evaluate_model.py --model <HF_REPO> [--steer <religion>]
```

Recognized religion values for steer: 

- `hindu` 
- `muslim`
- `christian` 
- `sikh`
- `buddhist` 
- `jain`


### SEA

```bash
python evaluate_model.py --model <HF_REPO> --country <COUNTRY>
```

Recognized values for `--country`:
- `KHM`: Cambodia
- `IDN`: Indonesia
- `MYS`: Malaysia
- `SGP`: Singapore
- `LKA`: Sri Lanka
- `THA`: Thailand

### EA

```bash
python evaluate_model.py --model <HF_REPO> --country <COUNTRY>
```

Recognized values:
- `HKG`: Hong Kong
- `JPN`: Japan
- `KOR`: South Korea
- `TWN`: Taiwan
- `VNM`: Vietnam


## Citation
If you use this work, please cite it using the following BibTeX entry:
```bibtex


```