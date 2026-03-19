# Useful terms and definitions

- __Tokenization__ - the process of breaking down text into smaller units called tokens, such as words, subwords, or characters. Tokenization relies on exact matches. Synonyms or related words are not recognized as similar. Tokens lack understanding of context and meaning.
- __Token__ - is a concept specific to LLMs and is core to LLM inference performance metrics. It is the unit, or smallest lingual entity, that LLMs use to break down and process natural language.

- __Embeddings__ - numerical representations of data. They convert complex, high-dimensional data into low-dimensional vectors. This transformation allows machines to process and analyze data efficiently. They capture sematic relationship unlike tokens.

- BERT (Bidirectional Encoder Representations from Transformers) -  generates contextual embeddings where the same word can have different embeddings based on its context. Token Limit: Standard BERT models have a maximum token limit of 512.
- __Input Sequence Length (ISL)__ - how many tokens that the LLM gets
- __Output Sequence Length (OSL)__ -  how many tokens the LLM generates.
- __Context Length__ - how many tokens the LLM uses at each generation step, including both the input and output tokens generated to that point. 
- __Inference performance__ how FAST the model responds
- __Reasoning performance__ = how WELL the model thinks
- __Ground truth__ - it is the verified, correct answer or reference standard used to evaluate AI system performance. Think of it as the "answer key" against which you measure your model's predictions.

- __Overfitting__ - occurs when a machine learning model learns training data too well, capturing noise and specific fluctuations rather than general patterns. This results in excellent performance on training data but poor, unreliable predictions on new, unseen data in production.
