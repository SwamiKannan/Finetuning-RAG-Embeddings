# Finetuning RAG embeddings

<p align="center">
<img src = "https://github.com/SwamiKannan/Finetuning-RAG-Embeddings/blob/main/images/cover.png"><br>
<sub> Base Image Credit: <a href="https://www.freepik.com/premium-photo/tuning-radio-radio-station_16035333.htm">Freepik</a> . Editing by <a href="https://github.com/SwamiKannan">Swaminathan Kannan</a></sub>
</p>

## Introduction
Fine tuning embeddings is super important when the distribution of the data you want to process is different from the typical distribution of data on which the original embedding model is trained on. Basically, if the data in your vectorstore is science and research-related but the data that was used for training the embeddings is business related, then similar words may have different connotations e.g. "relative" in physics may refer to Einstein's theory of relativity but in business, it may be more relevant to content on family businesses. Hence, for specialized topics, it makes sense to further fine tune the embeddings to re-orient the relation of words to each other.
Hence, Fine-tuning is the process of adjusting your embedding model to better fit the domain of your data. Though before fine-tuning it yourself, you should always take a look at the <a href="https://huggingface.co/models"> Hugging Face model database and check if someone already fine-tuned an embedding model on data that is similar to yours.

## Repo construct
### Converting dataset into the appropriate JSON structure
1. <b> If you have your own question - response / question - answer dataset</b>
    You can peruse <a href="https://huggingface.co/docs/datasets/index">Huggingface's datasets</a> to check if any of the datasets are relevant to your own domain.
    Follow the instructions as described <a href="https://github.com/SwamiKannan/Creating-Llamaindex-EmbeddingQAFinetuneDataset-compatible-files/blob/main/README.md"> here </a>

2. <b> Create a custom question answer dataset from your content </b>
    This requires an LLM such as OpenAI to generate custom question-reference sets.
   Follow the instructions as described <a href="https://docs.llamaindex.ai/en/latest/examples/finetuning/embeddings/finetune_embedding.html#generate-corpus">here</a>
   
### Finetuning the model
The function final_finetune() contains two sub-functions:
1. **create_json_input()** (which converts the input JSON into a format that can be easily uploaded into the EmbeddingQAFinetuneDataset object
2. **finetune()** which tunes the embedding model based in the data outputted by create_json_input()
While converting the dataset to the appropriate JSON structure, we return the train_path (path for the training split), the test_path (path for the test split) and the main_path (the parent path for storing the final model). Hence, using these three parameters, run 
```
final_finetune(train_path, os.path.join('..', main_dir, FILE_PATH), CORPUS_COUNT, MODEL_ID, MODEL_OUTPUT)
```
where: <br>
FILE_PATH = path w

### Testing the model
< WIP>
<br />

## Usage:
### Install
```
pip install llama-index-finetuning
pip install llama-index-embeddings-openai
```
### TRAINING
### Update
1. Navigate to the src folder
    ```
    cd src
    ```
2. Update the details in main.py
   ```
   # *************************** UPDATE ALL PARAMETERS BELOW *************************************************

    # Path for the folder where you want all the data and embeddings to be stored
    MAIN_PATH = '..//camelai_physics' 

    # Based embedding model
    MODEL_ID = "BAAI/bge-small-en-v1.5"

    # Dataset to be trained on
    DATASET = 'camel-ai/physics'

    # Corpus_count is basically used to mix multiple outputs including the correct response to help tune the embeddings model
    CORPUS_COUNT = 5

    # Output model name
    MODEL_OUTPUT = 'embeddings_camelai'

    # If only train set is present, then this ratio splits the data into train and test based on the ratio provided
    TEST_RATIO = 0.2

    # If the dataset has multiple splits, then you can specify all the splits that you want to process here.
    SPLITS = ['train']

    # No. of epochs to train
    EPOCHS = 3

    # Transform function to ensure that the question is stored in df['question'] and the response is stored in df['answer]
    def transform_df(df):
        df['answer'] = df['topic;'] + ' ' + df['sub_topic'] + ' '+df['message_2']
        df['question'] = df['message_1']


    # creating a model dictionary as {reference_model_name : (trained_model_name, folder_name)}
    model_dict = {MODEL_ID: (MODEL_OUTPUT, MAIN_PATH)}  # Just a general structure used if we want to train multiple models on the same dataset

### Run
```
cd..
python src/main.py 
```
## Output
```
MAIN_PATH
└───data
    ├───dataset
    │   └───DATASET
    ├───embeddings
    │   └───MODEL_OUTPUT
    │       ├───1_Pooling
    │       ├───2_Normalize
    │       └───eval
    └───logs
        └───train
```
