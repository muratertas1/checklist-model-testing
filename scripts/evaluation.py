import os


def evaluate_mft(file_path):
    """
    Evaluates Minimum Functionality Tests (MFT) by comparing system labels to gold labels for each sentence in the dataset.
    Parameters:
    - file_path (str): The path to the file containing the dataset with system predictions.
    Returns:
    - tuple: Contains the failure rate (as a percentage) and a list of sentence IDs that failed the MFT.
    """
    # Open the file in read mode with UTF-8 encoding
    with open(file_path, 'r', encoding='utf-8') as file:
        # Split the file into sentences
        sentences = file.read().split('\n\n')
        # Count the sentences, adjusting for a trailing empty string if present
        total_sentences = len(sentences) - 1 if sentences[-1] == '' else len(sentences)
        # Initialize counters for failed sentences and a list for their IDs
        failed_sentences = 0
        failed_sentence_ids = []

        # Iterate over each sentence
        for sentence in sentences:
            # Skip empty lines
            if sentence == '':
                continue
            # Split the sentence into lines
            lines = sentence.split('\n')
            # Get the sentence ID from the first line
            sentence_id = lines[0].split('\t')[0]
            # Get the gold labels and system labels for each line, ignoring lines with a gold label of '_'
            gold_labels = [line.split('\t')[3] for line in lines if line.split('\t')[3] != '_']
            system_labels = [line.split('\t')[4] for line in lines if line.split('\t')[3] != '_']
            # If the gold labels and system labels do not match, increment the failed sentences counter and add the sentence ID to the list of failed sentences
            if gold_labels != system_labels:
                failed_sentences += 1
                failed_sentence_ids.append(sentence_id)

    # Calculate the failure rate as a percentage
    failure_rate = (failed_sentences / total_sentences) * 100
    # Return the failure rate and the list of failed sentence IDs
    return failure_rate, failed_sentence_ids
def evaluate_inv(file_path):
    """
    Evaluates Invariance tests (INV) by comparing pairs of sentences to ensure that system labels for tokens with relevant gold labels (not '_') do not change between the sentence pairs regardless of perturbations.
    Parameters:
    - file_path (str): The path to the file containing the dataset with system predictions.
    Returns:
    - tuple: Contains the failure rate (as a percentage) and a list of sentence pair IDs that failed the INV test.
    """
    sentences_data = {}
    total_pairs = 0
    failed_pairs = 0
    failed_sentence_ids = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip() == "":
                continue

            sentence_id, token_id, token, gold_label, system_label = line.strip().split('\t')
            sentence_id = int(sentence_id)
            
            # Skip tokens with gold label '_'
            if gold_label == "_":
                continue
            
            # If it's a new sentence ID, initialise it with an empty list
            if sentence_id not in sentences_data:
                sentences_data[sentence_id] = []
            
            # Store both gold and system labels for tokens with gold labels other than '_'
            sentences_data[sentence_id].append((gold_label, system_label))

        sentence_ids = sorted(sentences_data.keys())
        # Making sure that the number of input sentences is even
        if len(sentence_ids) % 2 != 0: 
            print('Warning: uneven number of sentences!')

        # Evaluating sentences in pairs
        for i in range(0, len(sentence_ids), 2):
            first_sentence_data = sentences_data[sentence_ids[i]]
            second_sentence_data = sentences_data[sentence_ids[i+1]]
            total_pairs += 1
            pair_failed = False

            # Map gold labels to system labels for easier comparison
            first_sentence_map = {gold: system for gold, system in first_sentence_data}
            second_sentence_map = {gold: system for gold, system in second_sentence_data}

            # Iterate over tokens in the first sentence
            for gold_label, system_label in first_sentence_data:
                # Check if the same gold label exists in the second sentence and focus on comparing system labels
                if gold_label in second_sentence_map and system_label != second_sentence_map[gold_label]:
                    pair_failed = True # Flag it as true if there is a change in system labels
                    break
            
            if pair_failed: # Update counter and store failed sentence IDs.
                failed_pairs += 1
                failed_sentence_ids.extend([sentence_ids[i], sentence_ids[i+1]])

    failure_rate = ((failed_pairs / total_pairs) * 100) / 2
    return failure_rate, failed_sentence_ids



def evaluate_all_files(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if 'MFT' in filename:
            failure_rate, failed_sentence_ids = evaluate_mft(file_path)
            print(f"File: {filename}, Failure Rate: {failure_rate}%, Failed Sentence IDs: {failed_sentence_ids}")
        elif 'INV' in filename:
            failure_rate, failed_sentence_ids = evaluate_inv(file_path)
            print(f"File: {filename}, Failure Rate: {failure_rate}%, Failed Sentence IDs: {failed_sentence_ids}")

# Call the function
evaluate_all_files('../predictions')

