# KV-Caching-Implementation
Simple Implementation of KV-Caching 

# GPT-2 KV Cache Implementation Comparison

This repository contains code demonstrating and comparing two approaches to implementing text generation with GPT-2, focusing on the use of Key-Value (KV) caching for performance optimization.

## Files

-   `gpt2_kv_cache_transformers.py`: Implements GPT-2 generation using the `transformers` library from Hugging Face.
-   `gpt2_kv_cache_scratch.py`: Implements a GPT-2 model from scratch, including the KV cache mechanism.

## Overview

This project compares the performance of text generation with and without KV caching in two scenarios:

1.  **Using Hugging Face's `transformers` Library:** This approach leverages the pre-built `GPT2LMHeadModel` and `GPT2Tokenizer` for ease of use and efficiency.
2.  **Implementing GPT-2 from Scratch:** This approach provides a deeper understanding of the model's architecture and the KV cache mechanism.

## Running the Code

### Using Hugging Face Transformers

1.  **Installation:**
    ```bash
    pip install transformers torch matplotlib numpy
    ```
2.  **Run the script:**
    ```bash
    python gpt2_kv_cache_transformers.py
    ```

### Implementing GPT-2 from Scratch

1.  **Installation:**
    ```bash
    pip install torch matplotlib numpy
    ```
2.  **Run the script:**
    ```bash
    python gpt2_kv_cache_scratch.py
    ```

## Output

Both scripts will:

-   Generate text using GPT-2 with and without KV caching.
-   Print the generated text, total generation time, and average time per token for both methods.
-   Calculate and print the speedup factor achieved by using KV caching.
-   Generate and save two plots:
    -   `token_generation_times.png`: Visualizes the generation time per token for both methods.
    -   `cumulative_generation_times.png`: Visualizes the cumulative generation time for both methods.
-   For the scratch implementation, also generates `token_generation_times_scratch_top_bottom.png`.

## Key Differences: Hugging Face vs. Scratch Implementation

### Hugging Face (`transformers` Library)

-   **Abstraction:** Provides a high-level API that abstracts away the complexities of the model's architecture and implementation.
-   **Ease of Use:** Simplifies the process of loading pre-trained models and tokenizers.
-   **Efficiency:** Leverages optimized implementations for performance.
-   **Flexibility:** Allows for easy customization and fine-tuning.
-   **Ready to use:** The KV cache functionality is already implemented, only the use\_cache flag needs to be set.
-   **Standardized:** Follows well established conventions for model loading and use.

### Scratch Implementation

-   **Understanding:** Requires a deep understanding of the GPT-2 architecture and the KV cache mechanism.
-   **Control:** Provides complete control over the model's implementation.
-   **Educational Value:** Offers a valuable learning experience for understanding the inner workings of transformer models.
-   **Customization:** Allows for fine-grained customization of the model's behavior.
-   **Development time:** Takes much longer to develop and debug.
-   **Potential for errors:** Many points of failure are possible during the development process.
-   **No optimizations:** The model will run slower than the optimized hugging face implementation.

## Conclusion

Using the `transformers` library offers a convenient and efficient way to work with pre-trained models like GPT-2. Implementing the model from scratch provides a deeper understanding of the underlying mechanisms but requires more effort.

KV caching significantly improves the performance of text generation in both approaches by reducing redundant computations.
