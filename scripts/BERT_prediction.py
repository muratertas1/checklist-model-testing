from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import os

# Define a dictionary to map labels to indices
label_to_index = {'ARG0': 0, 'ARG1': 1, 'ARG1-DSP': 2, 'ARG2': 3, 'ARG3': 4, 'ARG4': 5, 'ARG5': 6, 'ARGA': 7,
                      'ARGM-ADJ': 8, 'ARGM-ADV': 9, 'ARGM-CAU': 10, 'ARGM-COM': 11, 'ARGM-CXN': 12, 'ARGM-DIR': 13,
                      'ARGM-DIS': 14, 'ARGM-EXT': 15, 'ARGM-GOL': 16, 'ARGM-LOC': 17, 'ARGM-LVB': 18, 'ARGM-MNR': 19,
                      'ARGM-MOD': 20, 'ARGM-NEG': 21, 'ARGM-PRD': 22, 'ARGM-PRP': 23, 'ARGM-PRR': 24, 'ARGM-REC': 25,
                      'ARGM-TMP': 26, 'C-ARG0': 27, 'C-ARG1': 28, 'C-ARG1-DSP': 29, 'C-ARG2': 30, 'C-ARG3': 31,
                      'C-ARG4': 32, 'C-ARGM-ADV': 33, 'C-ARGM-COM': 34, 'C-ARGM-CXN': 35, 'C-ARGM-DIR': 36,
                      'C-ARGM-EXT': 37, 'C-ARGM-GOL': 38, 'C-ARGM-LOC': 39, 'C-ARGM-MNR': 40, 'C-ARGM-PRP': 41,
                      'C-ARGM-PRR': 42, 'C-ARGM-TMP': 43, 'R-ARG0': 44, 'R-ARG1': 45, 'R-ARG2': 46, 'R-ARG3': 47,
                      'R-ARG4': 48, 'R-ARGM-ADJ': 49, 'R-ARGM-ADV': 50, 'R-ARGM-CAU': 51, 'R-ARGM-COM': 52,
                      'R-ARGM-DIR': 53, 'R-ARGM-GOL': 54, 'R-ARGM-LOC': 55, 'R-ARGM-MNR': 56, 'R-ARGM-TMP': 57, '_': 58}
index_to_label = {v: k for k, v in label_to_index.items()}


def determine_label(subtoken_labels):
    """
    Determine the most common label among the subtoken labels.

    Args:
        subtoken_labels (list): List of labels for the subtokens.

    Returns:
        The most common label among the subtoken labels.
    """
    return max(set(subtoken_labels), key=subtoken_labels.count)

def bert_e2e(model, tokenizer, index_to_label, input_file_path, output_file_path):
    """
    Perform end-to-end prediction using a BERT model.

    Args:
        model (AutoModelForTokenClassification): The BERT model.
        tokenizer (AutoTokenizer): The tokenizer.
        index_to_label (dict): Mapping from indices to labels.
        input_file_path (str): Path to the input file.
        output_file_path (str): Path to the output file.
        This function is created by using ChatGPT4 with prompting
        "Create a function that takes a BERT model, a tokenizer, a dictionary mapping indices to labels,
        an input file path, and an output file path, and performs end-to-end prediction using the BERT model
        on the input file, writing the output to the output file." and modified manually in accordance with the specific needs.
    """
    # Read sentences and their gold labels from the input file
    sentence_list, gold_list = read_sentences_from_file(input_file_path)

    # Open the output file in write mode with UTF-8 encoding
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        # Initialize sentence ID
        sentence_id = 1
        # Iterate over each sentence and its corresponding gold labels
        for sentence, gold_labels in zip(sentence_list, gold_list):
            # Tokenize the sentence and prepare it for the model
            inputs = tokenizer(sentence, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True,
                               return_offsets_mapping=True)
            # Get the offset mapping from the inputs and remove it from the inputs
            offset_mapping = inputs.pop('offset_mapping')

            # Set the model to evaluation mode
            model.eval()
            # Disable gradient calculation
            with torch.no_grad():
                # Feed the inputs to the model and get the outputs
                outputs = model(**inputs)
            # Get the logits from the outputs
            logits = outputs.logits
            # Get the predicted labels by finding the maximum value in the logits
            predictions = torch.argmax(logits, dim=-1)

            # Initialize a list to hold the labels for each word in the sentence
            word_labels = [None] * len(sentence)
            # Initialize the current word index
            current_word_index = 0
            # Initialize a list to hold the labels for each subtoken in a word
            subtoken_labels = []

            # Iterate over each subtoken, its offset, and its predicted label
            for idx, ((start, end), pred) in enumerate(zip(offset_mapping[0], predictions[0])):
                # If the start and end offsets are both 0, skip this subtoken
                if start == end == 0:
                    continue

                # Get the label corresponding to the predicted index
                label = index_to_label[pred.item()]

                # If the start offset is 0 and there are already some subtoken labels,
                # determine the label for the current word, increment the word index, and reset the subtoken labels
                if start == 0 and subtoken_labels:
                    word_labels[current_word_index] = determine_label(subtoken_labels)
                    current_word_index += 1
                    subtoken_labels = []
                # Add the label for the current subtoken to the list of subtoken labels
                subtoken_labels.append(label)

            # If there are any remaining subtoken labels, determine the label for the current word
            if subtoken_labels:
                word_labels[current_word_index] = determine_label(subtoken_labels)

            # Iterate over each token, its system label, and its gold label
            for token_id, (token, system_label, gold_label) in enumerate(zip(sentence, word_labels, gold_labels)):
                # If the token is "[SEP]", break the loop
                if token == "[SEP]":
                    break
                # If the token starts with "[PRED] ", remove this prefix
                if token[0:7] == "[PRED] ":
                    token = token[7:]
                    # Write the sentence ID, token ID, token, gold label, and system label to the output file
                output_file.write(f"{sentence_id}\t{token_id+1}\t{token}\t{gold_label}\t{system_label}\n")
            # Write an empty line to the output file to separate sentences
            output_file.write("\n")  # Separate sentences with an empty line
            # Increment the sentence ID
            sentence_id += 1

def read_sentences_from_file(file_path):
    """
    Read sentences and their gold labels from a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        A tuple of two lists: The first list contains the sentences, and the second list contains the gold labels.
    """
    # Initialize empty lists for sentences, gold labels, current sentence tokens, and current sentence gold labels
    sentences = []
    gold_list = []
    current_sentence_tokens = []
    current_sentence_gold = []

    # Open the file in read mode with UTF-8 encoding
    with open(file_path, 'r', encoding='utf-8') as file:
        # Iterate over each line in the file
        for line in file:
            # If the line is empty (i.e., it's the end of a sentence)
            if line.strip() == "":
                # If there are tokens in the current sentence
                if current_sentence_tokens:
                    # Add the current sentence tokens and gold labels to their respective lists
                    sentences.append(current_sentence_tokens)
                    gold_list.append(current_sentence_gold)
                    # Reset the current sentence tokens and gold labels
                    current_sentence_tokens = []
                    current_sentence_gold = []
            else:
                # If the line is not empty, it contains a token and its gold label
                # Split the line on tabs and get the token and gold label
                token, gold = line.strip().split('\t')[2:4]
                # Add the token and gold label to their respective lists
                current_sentence_tokens.append(token)
                current_sentence_gold.append(gold)

        # If there are tokens left in the current sentence after reading the entire file
        if current_sentence_tokens:
            # Add the remaining tokens and gold labels to their respective lists
            sentences.append(current_sentence_tokens)
            gold_list.append(current_sentence_gold)

    # Return the sentences and gold labels
    return sentences, gold_list

def process_all_files(input_directory, output_directory, model, tokenizer, index_to_label, version):
    """
        Process all files in a directory using a BERT model.

        Args:
            input_directory (str): Path to the input directory.
            output_directory (str): Path to the output directory.
            model (AutoModelForTokenClassification): The BERT model.
            tokenizer (AutoTokenizer): The tokenizer.
            index_to_label (dict): Mapping from indices to labels.
            version (int): Version number.
        """
    for filename in os.listdir(input_directory):
        if filename.endswith(f'V{version}.conllu'):
            input_file_path = os.path.join(input_directory, filename)
            output_file_path = os.path.join(output_directory, filename)
            bert_e2e(model, tokenizer, index_to_label, input_file_path, output_file_path)


model_path1 = "/home/mumu/VU/PGRD/playground/BERT1_new"
model_path2 = "/home/mumu/VU/PGRD/playground/BERT2_new"
model_path3 = "/home/mumu/VU/PGRD/playground/BERT3_new"

# Define your models
model1 = AutoModelForTokenClassification.from_pretrained(model_path1)
model2 = AutoModelForTokenClassification.from_pretrained(model_path2)
model3 = AutoModelForTokenClassification.from_pretrained(model_path3)
tokenizer1 = AutoTokenizer.from_pretrained(model_path1)
tokenizer2 = AutoTokenizer.from_pretrained(model_path2)
tokenizer3 = AutoTokenizer.from_pretrained(model_path3)

os.makedirs('../predictions', exist_ok=True)
# Process files for each model
process_all_files('../dataset/bert_input', '../predictions', model1, tokenizer1, index_to_label, 1)
process_all_files('../dataset/bert_input', '../predictions', model2, tokenizer2, index_to_label, 2)
process_all_files('../dataset/bert_input', '../predictions', model3, tokenizer3, index_to_label, 3)