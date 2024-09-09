import json
import os
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def preprocess_V1(json_file, output_directory):
    # Open the hierarchical JSON file and load the data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Iterate over each capability in the data
    for capability in data:
        # Iterate over each test_type in the capability
        for test_type in data[capability]:
            # Initialize an empty list for the CoNLL-U format output
            conllu_output = []
            # Iterate over each item in the test_type
            for item in data[capability][test_type]:
                # Get the 'sentence_id', 'sentence', and 'predicate_id' fields from the item
                sentence_id = item['sentence_id']
                sentence = item['sentence']
                predicate_id = item['predicate_id']

                # Tokenize the sentence using NLTK
                tokens = word_tokenize(sentence)
                # Get the predicate from the tokens using the 'predicate_id'
                predicate = tokens[predicate_id - 1]

                # Prepare the CoNLL-U format output
                for i, token in enumerate(tokens, start=1):
                    # Initialize the label as '_' (the default label if the token doesn't match any 'tokenX' field in the item)
                    label = '_'
                    # For each key in the item
                    for key in item.keys():
                        # If the key starts with 'token' and the token matches the value of the key
                        if key.startswith('token') and token == item[key]:
                            # Replace 'token' with 'expected' in the key to get the label key
                            label_key = key.replace('token', 'expected')
                            # If the label key is in the item, get the label
                            if label_key in item:
                                label = item[label_key]
                                break  # Stop searching if the token is found

                    # Append the 'sentence_id', token index, token, and label to the output
                    conllu_output.append(f"{sentence_id}\t{i}\t{token}\t{label}")

                # Add the '[SEP]' token and the predicate after the sentence
                conllu_output.append(f"{sentence_id}\t{i+1}\t[SEP]\t_")
                conllu_output.append(f"{sentence_id}\t{i+2}\t{predicate}\t_")
                # Add an empty line between sentences
                conllu_output.append('')

            # Write the output to a file named '{capability}_{test_type}_V1.conllu'
            output = os.path.join(output_directory, f'{capability}_{test_type}_V1.conllu')
            with open(output, 'w') as f:
                f.write('\n'.join(conllu_output))

def preprocess_V2(json_file, output_directory):
    """
    Preprocesses the given JSON file into a CoNLL-U format for version 2.

    Parameters:
    - json_file (str): The path to the JSON file to preprocess.

    The function operates as follows:
    - Opens the JSON file and loads the data.
    - Iterates over each capability in the data.
    - For each capability, it iterates over each test_type.
    - For each test_type, it iterates over each item.
    - For each item, it tokenizes the 'sentence' field using NLTK, and prepares a CoNLL-U format output.
    - Writes the output to a file named '{capability}_{test_type}_V2.conllu'.
    """
    # Open the hierarchical JSON file and load the data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Iterate over each capability in the data
    for capability in data:
        # Iterate over each test_type in the capability
        for test_type in data[capability]:
            # Initialize an empty list for the CoNLL-U format output
            conllu_output = []
            # Iterate over each item in the test_type
            for item in data[capability][test_type]:
                # Get the 'sentence_id', 'sentence', and 'predicate_id' fields from the item
                sentence_id = item['sentence_id']
                sentence = item['sentence']
                predicate_id = item['predicate_id']

                # Tokenize the sentence using NLTK
                tokens = word_tokenize(sentence)
                # Get the predicate from the tokens using the 'predicate_id'
                predicate = tokens[predicate_id - 1]

                # Find the position of the predicate in the tokens
                predicate_index = tokens.index(predicate) + 1

                # Prepare CoNLL-U format output
                for i, token in enumerate(tokens, start=1):
                    # Initialize the label as '_' (the default label if the token doesn't match any 'tokenX' field in the item)
                    label = '_'
                    # For each key in the item
                    for key in item.keys():
                        # If the key starts with 'token' and the token matches the value of the key
                        if key.startswith('token') and token == item[key]:
                            # Replace 'token' with 'expected' in the key to get the label key
                            label_key = key.replace('token', 'expected')
                            # If the label key is in the item, get the label
                            if label_key in item:
                                label = item[label_key]
                                break  # Stop searching if the token is found

                    # Append the 'sentence_id', token index, token, and label to the output
                    conllu_output.append(f"{sentence_id}\t{i}\t{token}\t{label}")

                # Add [SEP] token
                conllu_output.append(f"{sentence_id}\t{i+1}\t[SEP]\t_")

                # Add preceding token, predicate, and token after predicate
                for j in range(predicate_index - 2, predicate_index + 1):
                    if j >= 0 and j < len(tokens):
                        token = tokens[j]
                        label = '_'
                        for key in item.keys():
                            if key.startswith('token') and token == item[key]:
                                label_key = key.replace('token', 'expected')
                                if label_key in item:
                                    label = item[label_key]
                                    break
                        conllu_output.append(f"{sentence_id}\t{i+2}\t{token}\t{label}")
                        i += 1

                conllu_output.append('')  # Add an empty line between sentences

            # Write output to a file
            output = os.path.join(output_directory, f'{capability}_{test_type}_V2.conllu')
            with open(output, 'w') as f:
                f.write('\n'.join(conllu_output))

def preprocess_V3(json_file, output_directory):
    """
    Preprocesses the given JSON file into a CoNLL-U format for version 3.

    Parameters:
    - json_file (str): The path to the JSON file to preprocess.

    The function operates as follows:
    - Opens the JSON file and loads the data.
    - Iterates over each capability in the data.
    - For each capability, it iterates over each test_type.
    - For each test_type, it iterates over each item.
    - For each item, it tokenizes the 'sentence' field using NLTK, and prepares a CoNLL-U format output.
    - Writes the output to a file named '{capability}_{test_type}_V3.conllu'.
    """
    # Open the hierarchical JSON file and load the data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Iterate over each capability in the data
    for capability in data:
        # Iterate over each test_type in the capability
        for test_type in data[capability]:
            # Initialize an empty list for the CoNLL-U format output
            conllu_output = []
            # Iterate over each item in the test_type
            for item in data[capability][test_type]:
                # Get the 'sentence_id', 'sentence', and 'predicate_id' fields from the item
                sentence_id = item['sentence_id']
                sentence = item['sentence']
                predicate_id = item['predicate_id']

                # Tokenize the sentence using NLTK
                tokens = word_tokenize(sentence)
                predicate = tokens[predicate_id - 1]
                predicate_encountered = False

                # Prepare CoNLL-U format output
                for i, token in enumerate(tokens, start=1):
                    # Marking the predicate with the special token if it's the first matching token
                    if token == predicate and not predicate_encountered:
                        token = '[PRED] ' + token
                        predicate_encountered = True
                    label = '_'  # Default label if token doesn't match any tokenX
                    for key in item.keys():
                        if key.startswith('token') and token == item[key]:
                            label_key = key.replace('token', 'expected')
                            if label_key in item:
                                label = item[label_key]
                                break

                    # Append the 'sentence_id', token index, token, and label to the output
                    conllu_output.append(f"{sentence_id}\t{i}\t{token}\t{label}")
                conllu_output.append('')  # Add an empty line between sentences

            # Write output to a file
            output = os.path.join(output_directory, f'{capability}_{test_type}_V3.conllu')
            with open(output, 'w') as f:
                f.write('\n'.join(conllu_output))

def process_all_json_files(directory):
    # Create a new directory for the output files
    output_directory = os.path.join(directory, 'bert_input')
    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            json_file_path = os.path.join(directory, filename)
            # Change the output directory for the preprocess functions
            preprocess_V1(json_file_path, output_directory)
            preprocess_V2(json_file_path, output_directory)
            preprocess_V3(json_file_path, output_directory)

if __name__ == "__main__":
    datasets_directory = "../dataset"
    process_all_json_files(datasets_directory)