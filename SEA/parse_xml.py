import xml.etree.ElementTree as ET
import json

root = ET.parse('Pew South and Southeast Asia metadata.xml')
print(root.getroot())

questions = {}

for child in root.iter():
    if 'var' in child.tag and 'varFormat' not in child.tag:
        print(child.attrib['name'], end=' - ')
        questions[child.attrib['name']] = None
        
        child_dict = {}
        options = {}
        value = None
        for subchild in child:
            if 'labl' in subchild.tag:
                print(subchild.text)
                child_dict['question'] = subchild.text
                
            if 'catgry' in subchild.tag:
                for subsubchild in subchild:
                    if 'catValu' in subsubchild.tag:
                        value = subsubchild.text
                        print(subsubchild.text, end=' ')
                    if 'labl' in subsubchild.tag:
                        print(subsubchild.text)
                        options[value] = subsubchild.text
                child_dict['options'] = options
        questions[child.attrib['name']] = child_dict

with open('questions.json', 'w') as f:
    json.dump(questions, f, indent=4)