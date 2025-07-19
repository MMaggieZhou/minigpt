
from model import GPTModel
from train import train
from data import load_data, to_training_input_and_label
from execute import execute, generate
import torch
import torch.nn as nn

def test_model(): 
    batch_size = 2
    vocab_size = 10 
    dmodel = 8
    h = 2
    dk = 4
    dff = 16
    num_layers = 2
    seq_length = 5
    
    # bug 1: in correct initialization of x 
    x = torch.randint(0, 10, (batch_size, seq_length))
    gpt = GPTModel(vocab_size, dmodel, dk, h, dff, num_layers)
    output = gpt(x)
    print(output.size())  # should be (batch_size, seq_length, vocab_size)

def test_train(): 
    # A tiny feedforward network for regression
    class SimpleRegressor(nn.Module):
        def __init__(self, input_dim, hidden_dim=16):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, x):
            return self.net(x)

    # Generate some random data
    torch.manual_seed(42)
    N, D_in = 200, 5
    X = torch.randn(N, D_in)
    true_weights = torch.randn(D_in, 1)
    y = X @ true_weights + 0.1 * torch.randn(N, 1) 

    model = SimpleRegressor(input_dim=D_in)
    train(model, X, y, batch_size=32, epochs=100, lr=1e-2, weight_decay=1e-4, device='cpu')

    model.eval()
    with torch.no_grad():
        predictions = model(X)
        mse = nn.MSELoss()(predictions, y)
        print(f"Final training MSE: {mse.item():.4f}")

def test_data(): 
    # Test data loading and preprocessing
    # _, char_to_idx, _ = load_data("sample.txt")
    indices, char_to_idx, idx_to_char = load_data("甄嬛传剧本01-10.txt")
    print(f"Number of unique characters: {len(char_to_idx)}")
    print(f"number of characters in the text: {len(indices)}")

    

def test_execute():  
    model, encoder = execute(["sample.txt"], ".", seq_length=10, batch_size=2, epochs=2)
    generate(model, encoder, "第2幕", max_length=50)


def main():
    test_execute()

if __name__ == "__main__":
    main()