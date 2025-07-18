import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """Feed Forward Network as described in the paper."""
    # dmodel: embedding dimension, dff: feed forward dimension, dropout: dropout rate
    # The feed forward network consists of two linear transformations with a ReLU activation in between.
    # The dropout layer is applied after the first linear transformation.
    # The dropout layer helps prevent overfitting by randomly setting a fraction of the input units
    # to zero during training.


    # The feed forward network is applied independently to each position in the input sequence.
    # The feed forward network is applied to the output of the self-attention layer.


    def __init__(self, dmodel, dff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dmodel, dff)
        self.linear2 = nn.Linear(dff, dmodel)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x) 
        x = self.linear2(x)
        return x
    
# seems more complicated than Karpathy's implementation
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dmodel, dk, h):
      super().__init__()
      assert dmodel % h == 0, "Embedding dimension must be divisible by the number of heads"
      self.h = h
      self.dmodel = dmodel
      self.dk = dk
      self.dv = self.dmodel // self.h

      self.Q = nn.Linear(dmodel, dk * h)
      self.K = nn.Linear(dmodel, dk * h)
      self.V = nn.Linear(dmodel, dmodel)

    # break q, k, v into heads
    def _reshape(self, t):
        new_shape = t.size()[:-1] + (self.h, t.size()[-1] // self.h)
        t = t.view(new_shape) # (batch_size, sequence_l, h, dk or dv)
        return t.permute(0,2,1,3) # (batch_size, h, sequence_l, dk or dv)

    def forward(
        self,
        x,
        attention_mask=None, # all encoders share of a batch share the same mask and same applies to decoders
      ):
      B, T, _ = x.size() # B: batch size, T: sequence length
      # O(batch_size * sequence_l * dk * h * dmodel)
      Q = self._reshape(self.Q(x)) # (batch_size, h, sequence_l, dk)
      # O(batch_size * sequence_l * dk * h * dmodel)
      K = self._reshape(self.K(x)) # (batch_size, h, sequence_l, dk)
      # O(batch_size * sequence_l * dmodel^2)
      V = self._reshape(self.V(x)) # (batch_size, h, sequence_l, dv)

      # softmax(QK/dv-2)V, O(batch_size * h * dk * sequence_l ^ 2)
      scores = Q @ K.permute(0,1,3,2) / math.sqrt(self.dk) #(batch_size, h, sequence_l, sequence_l)
      if attention_mask != None:  # (batch_size, 1, sequence_l, sequence_l)
          scores = scores.masked_fill(attention_mask == 0, float('-inf'))
      probs = F.softmax(scores, dim=-1)
      # O(batch_size * dmodel * sequence_l^2)
      output = probs @ V # (batch_size, h, sequence_l, dv)

      # concat
      output = output.permute(0,2,1,3).contiguous() #? error view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
      return output.view(B, T, self.dmodel), scores # output: (batch_size, sequene_l, dmodel), return scores for debugging
     
class AttentionLayer(nn.Module):
    def __init__(self, dmodel, dk, h, dff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(dmodel, dk, h)
        self.feed_forward = FeedForward(dmodel, dff, dropout)
        self.norm1 = nn.LayerNorm(dmodel)
        self.norm2 = nn.LayerNorm(dmodel)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        # O(batch_size * sequence_l * dmodel)
        attn_output, _ = self.self_attention(x, attention_mask) # _ is the scores
        x = x + self.dropout1(attn_output) # residual connection
        x = x + attn_output # residual connection
        x = self.norm1(x) # layer normalization

        ff_output = self.feed_forward(x) # O(batch_size * sequence_l * dmodel)
        x = x + self.dropout2(ff_output) # residual connection
        x = x + ff_output
        x = self.norm2(x) # layer normalization
        return x
    
class GPTModel(nn.Module):
    """GPT Model with multiple layers of self-attention and feed-forward networks."""
    # vocab_size: size of the vocabulary, dmodel: embedding dimension, dk: dimension of the key vectors,
    # h: number of attention heads, dff: dimension of the feed-forward network,
    # num_layers: number of layers in the model, dropout: dropout rate
    def __init__(self, vocab_size, dmodel, dk, h, dff, num_layers, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.dmodel = dmodel
        self.embedding = nn.Embedding(vocab_size, dmodel)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1024, dmodel)) # max sequence length is 1024
        nn.init.normal_(self.positional_encoding, mean=0.0, std=0.02)
        self.layers = nn.ModuleList([
            AttentionLayer(dmodel, dk, h, dff, dropout) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(dmodel, vocab_size)

    def forward(self, x, attention_mask=None):
        # x: (batch_size, sequence_l), consists of token ids in the range of [0, vocab_size-1] 
        # attention_mask: (batch_size, 1, sequence_l, sequence_l)
        # print(f"Input shape: {x.shape}")  # Debugging line
        if attention_mask is None:
            seq_len = x.size(1)
            attention_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).view(1, 1, seq_len, seq_len)  # (1, seq_len, seq_len)
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        for layer in self.layers:
            x = layer(x, attention_mask)
        # potential bug: return logits instead of softmax probabilities 
        logits = self.output_layer(x) # (batch_size, sequence_l, vocab_size)
        return logits
    
    def generate(self, input_ids, max_length=50):
        for _ in range(max_length):
            seq_len = input_ids.shape[1]
            output = self.forward(input_ids)
            idx_next = torch.argmax(output[:, -1, :], dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, idx_next], dim=1)
        return input_ids