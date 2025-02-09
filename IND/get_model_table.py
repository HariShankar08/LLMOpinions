import json
import csv

demographic_cols = [
    'QRELSING', 'QGEN', 'QAGErec', 'QCASTE', 'QMARRIEDrec','QFERTrec', 'QCHGENabrec', 'Urban', 'REGION', 'QEDU', 'QINCINDrec'
]

with open('questions.json') as f:
    questions = json.load(f)    

files = ['llama3.1it.txt', 'mistralv0.3it.txt', 'llama3.2.txt', 'gemma2it.txt', 'llama3.1.txt', 'mistralv0.3.txt', 'gemma2.txt', 'llama3.2it.txt', 'mistralv0.1it.txt', 'mistralv0.1.txt']
short_names = ['L8BIt', 'M7Bv3It', 'L3B', 'G9BIt', 'L8B', 'M7Bv3', 'G9B', 'L3BIt', 'M7Bv1It', 'M7Bv1']

model_to_model_dict = {}
for file, short in zip(files, short_names):
    with open(file) as f:
        contents = f.read()
        contents = contents[contents.index('{'): contents.rindex('}')+1]  
    
    resp = contents.split('\n')
    current_question = None
    model_dict = {}
    option_dict = {}
    for line in resp:
        if current_question is None:
            current_question = line.split()[-1]
            continue

        if len(line.split()) in (0, 1):
            continue
        
        if not line[0].isdigit() and line[-1].isdigit():
            continue
        
        try:
            items = line.split(' ')
            first_num = int(items[0])
            last_num = int(items[-1])
            option_dict[first_num] = last_num
        except ValueError:
            model_dict[current_question] = option_dict
            current_question = line.split()[-1]
            option_dict = {}
    model_to_model_dict[short] = model_dict

short_in_order = ['L3B', 'L3BIt', 'L8B', 'L8BIt', 'M7Bv1', 'M7Bv1It', 'M7Bv3', 'M7Bv3It', 'G9B', 'G9BIt']

with open('model_table.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Demographic', 'Demographic Options', 'L3B', 'L3BIt', 'L8B', 'L8BIt', 'M7Bv1', 'M7Bv1It', 'M7Bv3', 'M7Bv3It', 'G9B', 'G9BIt'])
    print(demographic_cols)
    for col in demographic_cols:
        for option in sorted(questions[col]['options'], key=lambda x: int(x)):
            row = [col, questions[col]['options'][option]]
            for model in ['L3B', 'L3BIt', 'L8B', 'L8BIt', 'M7Bv1', 'M7Bv1It', 'M7Bv3', 'M7Bv3It', 'G9B', 'G9BIt']:
                model_dict = model_to_model_dict[model]
                if col in model_dict:
                    if int(option) in model_dict[col]:
                        row.append(model_dict[col][int(option)])
                    else:
                        row.append(0)
                else:
                    row.append(0)
            writer.writerow(row)