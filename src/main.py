import json
from json_utils import create_corpus, create_json
from argparse import ArgumentParser

####*****************ILLUSTRATIVE EXAMPLE OF INPUT FILE*****************###
    # This is only an example to show the data structure required by the create_corpus function. This section (till "ILLUSTRATIVE EXAMPLE ENDS" can be deleted or replaced by your own dict creation function)

def create_dataset():  
    with open('..//problems.json','r',encoding='utf-8') as f:
        dataset = json.load(f)

    preferred_subjects = ['units-and-measurement', 'science-and-engineering-practices', 'physics']

    dict_qa={}
    keys = list(dataset.keys())
    count_subj = 0
    count = 0
    for key in keys:
        vals = dataset[key]
        if vals['topic'] in preferred_subjects:
            question = vals['hint']+vals['question'] +''.join(vals['choices'])
            answer = vals['lecture'] + vals['solution'] if vals['solution']=="" else vals['lecture']
            dict_qa[count_subj]={'Question':question, 'Reference':answer}
            count_subj+=1
        count+=1

    print(f'{len(list(dict_qa.keys()))} question-answer sets created')
    print('Format for a sample key value pair:')
    for key, val in dict_qa.items():
        print(f'Key: {key}\nValue:{val}')
        break
    with open('qa_json.json','w', encoding='utf-8') as fo:
        json.dump(dict_qa,fo)

    ####*****************ILLUSTRATIVE EXAMPLE ENDS*****************###
        
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    
    filename = args.filename 

    with open(filename,'r', encoding='utf-8') as fi: #Name your input json file as qa_json.json or change the name pf the file in this line appropriately
        dict_sample = json.load(fi)
    content = create_corpus(dict_sample)
    create_json(content)
