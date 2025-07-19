from data import load_data, to_training_input_and_label
from model import GPTModel
from train import train
import torch

DEVICE = "cpu"  # Set device to 'cpu' or 'cuda' as needed
def set_device(device):
    """
    Set the device for PyTorch operations.
    Args:
        device (str): Device to set ('cpu' or 'cuda').
    """
    global DEVICE
    DEVICE = device

def execute(train_data_file, output_dir, dmodel=128, h=8, dk=64, dff=256, num_layers=6, seq_length=50, batch_size=32, epochs=10):  
    indices, encoder = load_data(train_data_file)
    X, Y = to_training_input_and_label(indices, seq_length, batch_size)
    
    # convert X and Y to tensors 
    #X = torch.tensor(X, device=DEVICE)
    #Y = torch.tensor(Y, device=DEVICE)
    vocab_size = encoder.vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
    gpt_model = GPTModel(vocab_size, dmodel, dk, h, dff, num_layers)
    # get number of parameters
    num_params = sum(p.numel() for p in gpt_model.parameters() if p.requires_grad)
    print(f"Number of parameters in the model: {num_params}")
    
    train(gpt_model, X, Y, output_dir, batch_size=batch_size, epochs=epochs, lr=1e-3, weight_decay=1e-4, device=DEVICE)
    return gpt_model, encoder

def generate(model, encoder, prompt, max_length=50):
    """
    # Generate text using the trained model
    Args:
        model (nn.Module): The trained model.
        sequence (str): The initial sequence to start generation.
        max_length (int): Maximum length of the generated sequence.
    Returns:
        str: Generated text.
    """
    model.eval()

    # bug: incorrect dimension and value
    if prompt is None or prompt == "":
        sequence = torch.zeros((1, 1), dtype=torch.long, device=DEVICE) # Start with a single token (e.g., index 0)
    else:
        # convert prompt to indices
        indices = encoder.encode(prompt)
    
        # convert indices to tensor
        sequence = torch.tensor(indices, dtype=torch.long, device=DEVICE).unsqueeze(0)
    sequence = model.generate(sequence, max_length=max_length)
    # convert sequence to chars  
    generated_indices = sequence[0].tolist()
    sentence = encoder.decode(generated_indices)
    print(f"Generated text: {sentence}")