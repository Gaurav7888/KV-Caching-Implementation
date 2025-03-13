import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
import matplotlib.pyplot as plt
import numpy as np

class GPT2WithKVCache:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()
        
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
                outputs = self.model(generated_ids)
                
            # Get the next token prediction
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            # Append the new token to the sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Track time for this token
            times_per_token.append(time.time() - token_start_time)
            
            # Early stopping if we generate an EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
        total_time = time.time() - start_time
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
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
            outputs = self.model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            
        # Get the first token prediction
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        
        # Start with the last token from the initial pass
        generated_ids = torch.cat([input_ids, next_token], dim=1)
        current_token = next_token
        
        for _ in range(max_new_tokens - 1):
            token_start_time = time.time()
            
            # We only need to process the new token, using the cached KV states
            with torch.no_grad():
                outputs = self.model(
                    current_token, 
                    past_key_values=past_key_values, 
                    use_cache=True
                )
                past_key_values = outputs.past_key_values
                
            # Get the next token prediction
            next_token_logits = outputs.logits[:, -1, :]
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
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return generated_text, total_time, times_per_token

def main():
    generator = GPT2WithKVCache()
    prompt = "Artificial intelligence is"
    max_tokens = 50
    
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
    plt.savefig('token_generation_times.png')
    
    # Plot cumulative times
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(all_times_no_cache), label='Without KV Cache')
    plt.plot(np.cumsum(all_times_with_cache), label='With KV Cache')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Cumulative Time (seconds)')
    plt.title('Cumulative Generation Time: With vs. Without KV Cache')
    plt.legend()
    plt.grid(True)
    plt.savefig('cumulative_generation_times.png')
    
    print("Plots saved as 'token_generation_times.png' and 'cumulative_generation_times.png'")

if __name__ == "__main__":
    main()