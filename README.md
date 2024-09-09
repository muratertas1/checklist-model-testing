Murat Ertas - Model Functionality Testing with CHECKLIST method

Included files and directory structure:
- exam_MURAT_ERTAS folder
    -dataset folder
        -bert_input folder (contains the input files for BERT)
        -challenge_dataset.json (the dataset for the test)

    -scripts folder
        -predictions folder (contains the predictions for the test)
        -BERT_prediction.py (the script for BERT prediction)
        -evaluation.py (the script for evaluation)
        -preprocess.py (the script for preprocessing)
    -requirements.txt

Instructions:

- To run the code, you need to have the required libraries installed. You can install the required libraries by running the following command in the terminal:
    - pip install -r requirements.txt

- The fine-tuned BERT models can be downloaded from the following links (both are the same, you can choose one of them):
    - https://drive.google.com/drive/folders/1Ml1bm-rHBeSFapnSEszUqA9EClUAVYRq?usp=sharing
    - https://www.dropbox.com/scl/fo/xv6pkmvqfs4eaptr0aw9i/h?rlkey=jk8ggqbkrngjclxduq2fod3dy&dl=0
    - After downloading the models, you need to link their paths in BERT_prediction.py script.


- To run the preprocessing script, you need to run the following command in the terminal:
    - python preprocess.py
    - This script will create the input files for BERT models in the bert_input folder.
    - This script reads json files from the 'dataset' folder and creates the input files for BERT.

- To run the BERT prediction script, you need to run the following command in the terminal:
    - python BERT_prediction.py
    - This script will create the predictions for the test in the predictions folder.
    - Downloaded model paths should be linked in the script.
    - This script reads the preprocessed files from the 'bert_input' folder and make predictions, then saves predictions in the 'predictions' folder.

- To evaluate the predictions, you need to run the following command in the terminal:
    - python evaluation.py
    - This script will evaluate the predictions and print the results.
    - This script reads the predictions from the 'predictions' folder and evaluates them.
