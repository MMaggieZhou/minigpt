import torch

def load_data(file_path):
    """
    Load and preprocess text data from a file.

    Args:
        file_path (str): Path to the text file.
    Returns:
        tuple: A tuple containing:
            - indices (list): List of indices corresponding to characters in the text.
            - char_to_idx (dict): Dictionary mapping characters to their indices.
            - idx_to_char (dict): Dictionary mapping indices back to characters.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        # this already includes newline
        text = list(text)
        print(len(text))
    # convert text from list to set 
    unique_chars = sorted(set(text)) # for consistent char_to_idx
    # assign each char to an index
    char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    # convert text to indices
    indices = [char_to_idx[char] for char in text]
    return indices, char_to_idx, idx_to_char

def to_training_input_and_label_(indices, seq_length, batch_size):
    """
    Convert a list of character indices into training input and labels for transformer model.
    Args:
        indices (list): List of character indices.
        seq_length (int): Length of each sequence.
        batch_size (int): Number of sequences in a batch.
    Returns:
        tuple: A tuple containing:
            - input_tensor (torch.Tensor): A tensor of shape (batch_size, seq_length) containing the training input.
            - labels_tensor (torch.Tensor): A tensor of shape (batch_size, seq_length) containing the labels.
    """
    # Create labels by shifting the indices
    labels = indices[1:]
    indices = indices[:-1]
    X = []
    Y = []
    for i in range(len(indices) - seq_length + 1):
        X.append(indices[i:i + seq_length])
        Y.append(labels[i:i + seq_length])

    # convert X and Y into batches of size batch_size
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    # Ensure that the number of sequences is a multiple of batch_size
    # incorrect 
    # X = torch.stack([X[i:i + batch_size] for i in range(0, len(X), batch_size)])
    # Y = torch.stack([Y[i:i + batch_size] for i in range(0, len(Y), batch_size)])
    
    # Truncate so total number of sequences is divisible by batch_size
    num_full_batches = len(X) // batch_size
    X = X[:num_full_batches * batch_size]
    Y = Y[:num_full_batches * batch_size]
    # shuffle X and Y
    perm = torch.randperm(X.size(0))
    X = X[perm]
    Y = Y[perm]

    # Reshape into batches
    X = X.view(num_full_batches, batch_size, seq_length)
    Y = Y.view(num_full_batches, batch_size, seq_length)

    return X, Y

def to_training_input_and_label(indices, seq_length, device="cpu"):
    """
    Generate a batch of input-target pairs from a continuous stream of token indices.

    Args:
        indices (torch.Tensor): 1D tensor of token indices (e.g., training or validation data).
        batch_size (int): Number of sequences in the batch.
        seq_length (int): Length of each sequence.
        device (str): Device to move tensors to ('cpu' or 'cuda').

    Returns:
        X (torch.Tensor): Input tensor of shape (batch_size, seq_length)
        Y (torch.Tensor): Target tensor of shape (batch_size, seq_length)
    """
    X = []
    Y = []

    for i in range(len(indices) - seq_length - 1):
        # Ensure we have enough indices to form a full sequence
        if i + seq_length + 1 <= len(indices):
            X.append(indices[i:i + seq_length])
            Y.append(indices[i + 1:i + seq_length + 1])
    # Slice out input and target sequences
    #x = torch.stack([indices[i:i + seq_length] for i in ix])
    #y = torch.stack([indices[i + 1:i + seq_length + 1] for i in ix])
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y