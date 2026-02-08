# Definitions

### Attention Mechanism

- The attention mechanism is how transformers (like GPT, Llama, etc.) process text

Simple Analogy
Imagine reading a sentence:
"The cat sat on the mat because it was tired"
💡 When you read "it", your brain automatically connects it back to "cat". That's what attention does - it creates connections between related words.

- Works within ONE inference call
- Limited by context window (e.g., 8K, 128K tokens)
- Recomputed fresh each call
- Part of the MODEL
