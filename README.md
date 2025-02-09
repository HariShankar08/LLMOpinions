## Directory Structure

```
.
├── EA
│   ├── Llama-3.1-8B
│   │   ├── HKG.txt
│   │   ├── JPN.txt
│   │   ├── KOR.txt
│   │   ├── TWN.txt
│   │   └── VNM.txt
│   ├── Llama-3.1-8B-Instruct
│   │   ├── HKG.txt
│   │   ├── JPN.txt
│   │   ├── KOR.txt
│   │   ├── TWN.txt
│   │   └── VNM.txt
│   ├── Llama-3.2-3B
│   │   ├── HKG.txt
│   │   ├── JPN.txt
│   │   ├── KOR.txt
│   │   ├── TWN.txt
│   │   └── VNM.txt
│   ├── Llama-3.2-3B-Instruct
│   │   ├── HKG.txt
│   │   ├── JPN.txt
│   │   ├── KOR.txt
│   │   ├── TWN.txt
│   │   └── VNM.txt
│   ├── Mistral-7B-Instruct-v0.1
│   │   ├── HKG.txt
│   │   ├── JPN.txt
│   │   ├── KOR.txt
│   │   ├── TWN.txt
│   │   └── VNM.txt
│   ├── Mistral-7B-Instruct-v0.3
│   │   ├── HKG.txt
│   │   ├── JPN.txt
│   │   ├── KOR.txt
│   │   ├── TWN.txt
│   │   └── VNM.txt
│   ├── Mistral-7B-v0.1
│   │   ├── HKG.txt
│   │   ├── JPN.txt
│   │   ├── KOR.txt
│   │   ├── TWN.txt
│   │   └── VNM.txt
│   ├── Mistral-7B-v0.3
│   │   ├── HKG.txt
│   │   ├── JPN.txt
│   │   ├── KOR.txt
│   │   ├── TWN.txt
│   │   └── VNM.txt
│   ├── Pew East Asian Societies metadata.xml
│   ├── evaluate.sh
│   ├── evaluate_model.py
│   ├── gemma-2-9b
│   │   ├── HKG.txt
│   │   ├── JPN.txt
│   │   ├── KOR.txt
│   │   ├── TWN.txt
│   │   └── VNM.txt
│   ├── gemma-2-9b-it
│   │   ├── HKG.txt
│   │   ├── JPN.txt
│   │   ├── KOR.txt
│   │   ├── TWN.txt
│   │   └── VNM.txt
│   ├── model_questions.py
│   ├── parse_xml.py
│   ├── questions.json
│   ├── responses.csv
│   └── sorted-scores.csv
├── IND
│   ├── Pew India DDI metadata.xml
│   ├── eval_steering.sh
│   ├── evaluate.sh
│   ├── evaluate_model.py
│   ├── evaluate_quantized_model.py
│   ├── get_model_table.py
│   ├── get_tables.py
│   ├── model_table.csv
│   ├── parse_xml.py
│   ├── questions.json
│   ├── responses.csv
│   ├── results
│   │   ├── gemma2.txt
│   │   ├── gemma2it.txt
│   │   ├── llama3.1.txt
│   │   ├── llama3.1it.txt
│   │   ├── mistralv0.1.txt
│   │   ├── mistralv0.1it.txt
│   │   ├── mistralv0.3.txt
│   │   └── mistralv0.3it.txt
│   └── steering
│       ├── buddhist.txt
│       ├── christian.txt
│       ├── hindu.txt
│       ├── jain.txt
│       ├── muslim.txt
│       ├── parsi.txt
│       ├── sikh.txt
│       ├── unsteered
│       └── unsteered.txt
├── README.md
└── SEA
    ├── Llama-3.1-8B
    │   ├── IDN.txt
    │   ├── KHM.txt
    │   ├── LKA.txt
    │   ├── MYS.txt
    │   ├── SGP.txt
    │   └── THA.txt
    ├── Llama-3.1-8B-Instruct
    │   ├── IDN.txt
    │   ├── KHM.txt
    │   ├── LKA.txt
    │   ├── MYS.txt
    │   ├── SGP.txt
    │   └── THA.txt
    ├── Llama-3.2-3B
    │   ├── IDN.txt
    │   ├── KHM.txt
    │   ├── LKA.txt
    │   ├── MYS.txt
    │   ├── SGP.txt
    │   └── THA.txt
    ├── Llama-3.2-3B-Instruct
    │   ├── IDN.txt
    │   ├── KHM.txt
    │   ├── LKA.txt
    │   ├── MYS.txt
    │   ├── SGP.txt
    │   └── THA.txt
    ├── Mistral-7B-Instruct-v0.1
    │   ├── IDN.txt
    │   ├── KHM.txt
    │   ├── LKA.txt
    │   ├── MYS.txt
    │   ├── SGP.txt
    │   └── THA.txt
    ├── Mistral-7B-Instruct-v0.3
    │   ├── IDN.txt
    │   ├── KHM.txt
    │   ├── LKA.txt
    │   ├── MYS.txt
    │   ├── SGP.txt
    │   └── THA.txt
    ├── Mistral-7B-v0.1
    │   ├── IDN.txt
    │   ├── KHM.txt
    │   ├── LKA.txt
    │   ├── MYS.txt
    │   ├── SGP.txt
    │   └── THA.txt
    ├── Mistral-7B-v0.3
    │   ├── IDN.txt
    │   ├── KHM.txt
    │   ├── LKA.txt
    │   ├── MYS.txt
    │   ├── SGP.txt
    │   └── THA.txt
    ├── Pew South and Southeast Asia metadata.xml
    ├── evaluate.sh
    ├── evaluate_model.py
    ├── gemma-2-9b
    │   ├── IDN.txt
    │   ├── KHM.txt
    │   ├── LKA.txt
    │   ├── MYS.txt
    │   ├── SGP.txt
    │   └── THA.txt
    ├── gemma-2-9b-it
    │   ├── IDN.txt
    │   ├── KHM.txt
    │   ├── LKA.txt
    │   ├── MYS.txt
    │   ├── SGP.txt
    │   └── THA.txt
    ├── model_questions.py
    ├── parse_xml.py
    ├── questions.json
    ├── responses.csv
    └── sorted-scores.csv

26 directories, 155 files
```