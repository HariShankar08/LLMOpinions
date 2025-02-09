import pandas as pd
import json
import csv


with open('questions.json') as f:
    q = json.load(f)

df = pd.read_csv('responses.csv')


demographic_cols = ['QRELSING', 'QGEN', 'QAGErec', 'QCASTE', 'QMARRIEDrec', 'QFERTrec', 'QCHGENabrec',
                    'Urban', 'REGION', 'QEDU', 'QINCINDrec'
                    ]

# Open a csv writer for the file 'india_demographics.csv'
with open('india_demographics.csv', 'w') as f:
    writer = csv.writer(f)
    for col in demographic_cols:
        value_counts = df[col].value_counts().to_dict()
        options = q[col]['options']
        for option in sorted(value_counts):
            writer.writerow([col, options[str(option)], value_counts.get(option, 0)])
        