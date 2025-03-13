import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import matplotlib.pyplot as plt
import numpy as np

class Config:
    """Configuration class for the model parameters."""
    def __init__(self):
        self.vocab_size = 50257  # GPT-2 vocabulary size
        self.n_positions = 1024  # Maximum sequence length
        self.n_embd = 768        # Embedding dimension (for small GPT-2)
        self.n_layer = 12        # Number of transformer layers
        self.n_head = 12         # Number of attention heads
        self.n_inner = 4 * 768   # Dimension of feed-forward layer
        self.activation_function = "gelu"
        self.resid_pdrop = 0.1   # Dropout probability
        self.embd_pdrop = 0.1    # Embedding dropout
        self.attn_pdrop = 0.1    # Attention dropout
        self.layer_norm_epsilon = 1e-5  # Layer norm epsilon

class LayerNorm(nn.Module):
    """Layer Normalization module."""
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias

class MLP(nn.Module):
    """Multi-layer perceptron for feed-forward network."""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)
        
    def gelu(self, x):
        """Gaussian Error Linear Unit activation function."""
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        
    def forward(self, x):
        h = self.gelu(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)

class Attention(nn.Module):
    """Multi-head attention mechanism."""
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = nn.Dropout(config.attn_pdrop)
        
        # Key, query, value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.head_dim = config.n_embd // config.n_head
        
    def split_heads(self, x):
        """Split the embedding dimension into multiple heads."""
        new_shape = x.size()[:-1] + (self.n_head, self.head_dim)
        x = x.view(*new_shape)
        # Rearrange to [batch, heads, sequence, head_dim]
        return x.permute(0, 2, 1, 3)
        
    def merge_heads(self, x):
        """Merge the multiple heads back into embedding dimension."""
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_shape)
        
    def forward(self, x, layer_past=None, use_cache=False):
        # [batch_size, seq_length, n_embd]
        batch_size, seq_length, _ = x.shape
        
        # Project input to query, key, value
        # [batch_size, seq_length, 3*n_embd]
        qkv = self.c_attn(x)
        
        # Split into query, key, value
        # Each: [batch_size, seq_length, n_embd]
        query, key, value = qkv.chunk(3, dim=2)
        
        # Split heads
        # Each: [batch_size, n_head, seq_length, head_dim]
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        
        # Handle KV caching
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
            
        present = (key, value) if use_cache else None
        
        # Compute attention scores
        # [batch_size, n_head, seq_length, key_length]
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)
        
        # Apply causal (triangular) mask
        mask = torch.tril(torch.ones(seq_length, key.size(2))).view(1, 1, seq_length, key.size(2))
        scores = scores.masked_fill(mask == 0, -1e10)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        # [batch_size, n_head, seq_length, head_dim]
        attn_output = torch.matmul(attn_weights, value)
        
        # Merge heads back together
        # [batch_size, seq_length, n_embd]
        attn_output = self.merge_heads(attn_output)
        
        # Project to output
        attn_output = self.c_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        return attn_output, present

class Block(nn.Module):
    """Transformer block with attention and MLP."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = Attention(config)
        self.ln_2 = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)
        
    def forward(self, x, layer_past=None, use_cache=False):
        # Self-attention with residual connection
        a, present = self.attn(self.ln_1(x), layer_past=layer_past, use_cache=use_cache)
        x = x + a
        
        # MLP with residual connection
        m = self.mlp(self.ln_2(x))
        x = x + m
        
        return x, present

class GPT2Model(nn.Module):
    """GPT-2 model implementation."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Transformer blocks
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
    def forward(self, input_ids, past_key_values=None, use_cache=False):
        batch_size, seq_length = input_ids.shape
        
        # Create position ids
        position_ids = torch.arange(0, seq_length, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Get embeddings
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        
        # Initialize past_key_values if None
        if past_key_values is None:
            past_key_values = [None] * len(self.h)
        
        # Initialize present_key_values for KV caching
        present_key_values = [] if use_cache else None
        
        # Process through transformer blocks
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            hidden_states, present = block(
                hidden_states, 
                layer_past=layer_past,
                use_cache=use_cache
            )
            
            if use_cache:
                present_key_values.append(present)
        
        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        return hidden_states, present_key_values

class GPT2LMHeadModel(nn.Module):
    """GPT-2 language model head."""
    def __init__(self, config):
        super().__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights between embedding and output layer
        self.lm_head.weight = self.transformer.wte.weight
        
    def forward(self, input_ids, past_key_values=None, use_cache=False):
        hidden_states, present_key_values = self.transformer(
            input_ids, 
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        
        # Get logits
        lm_logits = self.lm_head(hidden_states)
        
        return lm_logits, present_key_values

class SimplifiedTokenizer:
    """A very simplified tokenizer for demonstration purposes."""
    def __init__(self):
        # For a real implementation, you'd load actual vocabulary and tokens
        self.vocab_size = 50257
        self.eos_token_id = 50256
        
    def encode(self, text, return_tensors=None):
        """Simulate encoding by returning random token IDs."""
        # In a real implementation, you'd properly tokenize the text
        ids = torch.randint(0, 1000, (1, len(text.split())))
        if return_tensors == "pt":
            return ids
        return ids.tolist()
        
    def decode(self, ids, skip_special_tokens=True):
        """Simulate decoding by returning generic text."""
        # In a real implementation, you'd convert token IDs back to text
        # Just return a generic response for demonstration
        return "Generated text: " + " ".join(["word"] * len(ids.squeeze()))

class GPT2WithKVCache:
    def __init__(self):
        self.config = Config()
        self.model = GPT2LMHeadModel(self.config)
        self.tokenizer = SimplifiedTokenizer()
        
    def generate_without_kv_cache(self, prompt, max_new_tokens=50):
        """Generate text without using KV cache (naive approach)."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Track generation time
        start_time = time.time()
        times_per_token = []
        
        generated_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            token_start_time = time.time()
            
            # For each new token, we run the full sequence through the model
            with torch.no_grad():
                logits, _ = self.model(generated_ids, use_cache=False)
                
            # Get the next token prediction
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            # Append the new token to the sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Track time for this token
            times_per_token.append(time.time() - token_start_time)
            
            # Early stopping if we generate an EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
        total_time = time.time() - start_time
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text, total_time, times_per_token
    
    def generate_with_kv_cache(self, prompt, max_new_tokens=50):
        """Generate text using KV cache for efficiency."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Track generation time
        start_time = time.time()
        times_per_token = []
        
        # Initial forward pass to get the KV cache
        with torch.no_grad():
            # past_key_values will contain the key/value states
            logits, past_key_values = self.model(input_ids, use_cache=True)
            
        # Get the first token prediction
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        
        # Start with the last token from the initial pass
        generated_ids = torch.cat([input_ids, next_token], dim=1)
        current_token = next_token
        
        for _ in range(max_new_tokens - 1):
            token_start_time = time.time()
            
            # We only need to process the new token, using the cached KV states
            with torch.no_grad():
                logits, past_key_values = self.model(
                    current_token, 
                    past_key_values=past_key_values, 
                    use_cache=True
                )
                
            # Get the next token prediction
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            # Append the new token to the sequence
            current_token = next_token
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Track time for this token
            times_per_token.append(time.time() - token_start_time)
            
            # Early stopping if we generate an EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        total_time = time.time() - start_time
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text, total_time, times_per_token

def main():
    print("Initializing custom GPT-2 model from scratch...")
    generator = GPT2WithKVCache()
    prompt = "Artificial intelligence is"
    max_tokens = 20
    
    print(f"Prompt: {prompt}")
    print("\n--- Generating without KV cache ---")
    text_no_cache, time_no_cache, times_per_token_no_cache = generator.generate_without_kv_cache(prompt, max_tokens)
    
    print(f"Generated text:\n{text_no_cache}")
    print(f"Total generation time: {time_no_cache:.4f} seconds")
    print(f"Average time per token: {time_no_cache / len(times_per_token_no_cache):.4f} seconds")
    
    print("\n--- Generating with KV cache ---")
    text_with_cache, time_with_cache, times_per_token_with_cache = generator.generate_with_kv_cache(prompt, max_tokens)
    
    print(f"Generated text:\n{text_with_cache}")
    print(f"Total generation time: {time_with_cache:.4f} seconds")
    print(f"Average time per token: {time_with_cache / len(times_per_token_with_cache):.4f} seconds")
    
    print("\n--- Performance comparison ---")
    print(f"Speedup factor: {time_no_cache / time_with_cache:.2f}x")
    
    # Plotting
    # For without cache, add the first token time which isn't in the list
    if len(times_per_token_no_cache) > 0:
        first_token_time = time_no_cache - sum(times_per_token_no_cache)
        all_times_no_cache = [first_token_time] + times_per_token_no_cache
    else:
        all_times_no_cache = [time_no_cache]
        
    # For with cache, add the first token time
    if len(times_per_token_with_cache) > 0:
        first_token_time = time_with_cache - sum(times_per_token_with_cache)
        all_times_with_cache = [first_token_time] + times_per_token_with_cache
    else:
        all_times_with_cache = [time_with_cache]
    
    # Plot generation times
    plt.figure(figsize=(10, 6))
    plt.plot(all_times_no_cache, label='Without KV Cache')
    plt.plot(all_times_with_cache, label='With KV Cache')
    plt.xlabel('Token Position')
    plt.ylabel('Generation Time (seconds)')
    plt.title('Token Generation Time: With vs. Without KV Cache')
    plt.legend()
    plt.grid(True)
    plt.savefig('token_generation_times_scratch.png')
    
    # Plot cumulative times
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(all_times_no_cache), label='Without KV Cache')
    plt.plot(np.cumsum(all_times_with_cache), label='With KV Cache')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Cumulative Time (seconds)')
    plt.title('Cumulative Generation Time: With vs. Without KV Cache')
    plt.legend()
    plt.grid(True)
    plt.savefig('cumulative_generation_times_scratch.png')
    
    print("Plots saved as 'token_generation_times_scratch.png' and 'cumulative_generation_times_scratch.png'")

if __name__ == "__main__":
    main()