# Prompt Engineering for Agentic AI

## Overview

Prompt engineering is the practice of designing inputs to LLMs to get optimal outputs. For agents, it's critical - the system prompt defines agent behavior, the tool prompts control action selection, and evaluation prompts measure performance.

---

## Zero-Shot Prompting

**Definition:** Directly asking the LLM to perform a task without providing any examples or demonstrations.

The model relies entirely on its pre-training knowledge and the instructions in your prompt.

**Example:**

```
"Classify the sentiment of this review: 'This product is amazing!'"
→ Model infers it should output: positive/negative/neutral
```

**Advantages:**

- Quick and simple - no need to craft examples
- Works well for common tasks the model was trained on
- Minimal prompt engineering required

**Limitations:**

- May not work well for complex, domain-specific, or novel tasks
- Performance depends heavily on how well task was represented in training data
- Less control over output format and style

**Best for:** Classification, simple Q&A, well-known tasks, general knowledge queries

---

## Few-Shot Prompting

**Definition:** Providing the LLM with a few examples (typically 2-5) that demonstrate the desired input-output behavior before asking it to perform the task.

The model learns from these examples to understand the pattern and apply it to new inputs.

**Example format:**

```
Input: "I love this!" → Output: Positive
Input: "Terrible experience" → Output: Negative
Input: "It's okay" → Output: Neutral
Input: "Best purchase ever!" → Output: ?
```

**Advantages:**

- Significantly improves performance on specific tasks
- Provides clear examples of desired format and style
- Helps the model understand nuanced requirements
- More reliable for domain-specific tasks

**Considerations:**

- Examples should be diverse and representative
- Quality of examples directly impacts performance
- Uses more tokens (costs more, takes longer)
- Need to balance number of examples (more isn't always better)

**Best for:** Custom formats, domain-specific tasks, when consistency is critical, teaching new patterns

---

## Chain-of-Thought (CoT) Prompting

**Definition:** Encouraging the LLM to break down its reasoning process into intermediate steps before arriving at the final answer.

Can be combined with zero-shot ("Let's think step by step") or few-shot (showing reasoning examples).

**Zero-shot CoT example:**

```
Q: If a store has 15 apples and sells 40% of them, how many are left?
A: Let's think step by step:
1. 40% of 15 apples = 0.4 × 15 = 6 apples sold
2. Apples remaining = 15 - 6 = 9
```

**Few-shot CoT example:**

```
Q: 23 + 47 = ?
A: Let's break this down:
   23 + 47
   = 20 + 3 + 40 + 7
   = (20 + 40) + (3 + 7)
   = 60 + 10 = 70

Q: 156 + 278 = ?
A: Let's break this down:
   ...
```

**Advantages:**

- Dramatically improves performance on reasoning tasks
- Makes the model's reasoning transparent and verifiable
- Helps catch and correct errors in logic
- Particularly effective for math, logic, and multi-step problems

**Key techniques:**

- **Self-consistency:** Generate multiple reasoning paths and select the most common answer
- **Zero-shot CoT:** Simply add "Let's think step by step" to your prompt
- **Manual CoT:** Provide worked examples with reasoning steps

**Best for:** Mathematical reasoning, logical problems, multi-step tasks, debugging, complex decision-making

---

## ReAct (Reasoning + Acting)

**Definition:** Combines reasoning traces with task-specific actions in an interleaved manner.

Critical for agentic AI systems - allows agents to reason about what to do next and then take actions.

**Format pattern:**

```
Thought: [reasoning about current situation]
Action: [specific action to take]
Observation: [result of action]
Thought: [reasoning about observation]
... (repeat until task complete)
```

**Example:**

```
Thought: I need to find the current temperature in Paris
Action: search("Paris weather today")
Observation: High 22°C, Low 18°C, Sunny
Thought: Now I have the information needed
Action: respond("The weather in Paris is sunny with a high of 22°C")
```

**Use cases:** Tool use, web navigation, database queries, multi-step problem solving

**Key for agents:** Enables dynamic decision-making based on intermediate results

---

## Prompt Chaining

**Definition:** Breaking complex tasks into a sequence of smaller prompts where output of one becomes input to the next.

Each step can be optimized independently.

**Example workflow:**

```
Step 1: Extract entities → Output: [Entity List]
Step 2: Classify sentiment → Input: [Entity List] → Output: [Sentiment Labels]
Step 3: Generate summary → Input: [Entities, Sentiments] → Output: [Summary]
Step 4: Format output → Input: [Summary] → Output: [Final Response]
```

**Advantages:**

- Better control over each step
- Easier to debug and improve
- Can use different models/parameters for different steps
- Reduces context window requirements

**Important for agents:** Agents often chain prompts to decompose complex goals into manageable subtasks

---

## System Prompts vs User Prompts

**System prompts:** Set the overall behavior, role, and constraints for the AI

- Define personality, expertise, output format rules
- Persistent across conversation
- Example: "You are an expert Python developer who writes clean, well-documented code"

**User prompts:** Specific tasks or queries from the user

- Change with each interaction
- Build on the system prompt context
- Example: "Write a function to sort an array"

**For agents:** System prompts define agent capabilities, tools available, and operational guidelines

---

## Role Prompting

**Definition:** Assigning a specific role or persona to the LLM to improve output quality.

**Examples:**

- "Act as a senior software architect"
- "You are a financial analyst"
- "You are an expert in machine learning"

**How it works:** Leverages training data associated with that role's expertise

**More effective than generic prompts for domain-specific tasks**

**For agents:** Defines the agent's expertise and decision-making framework

---

## Tool Use Prompting

**Definition:** Explicitly instructing the model when and how to use external tools/APIs.

Common in agentic systems where LLM orchestrates multiple tools.

**Format:** Describe available tools, when to use them, expected input/output formats

**Function calling:** Structured way for models to request tool usage

**Example:**

```
Available tools:
- search(query): Search the web
- calculate(expression): Evaluate math expression
- get_time(timezone): Get current time

When to use:
- Use search for current information
- Use calculate for math problems
- Use get_time for timezone questions
```

**For agents:** Core capability - agents must know when to use calculators, databases, search engines, etc.

---

## Self-Reflection / Self-Critique

**Definition:** Prompting the model to evaluate and improve its own outputs.

**Pattern:** Generate → Critique → Refine → Output

**Example:**

```
Step 1 (Generate): "Based on the data, I think the answer is X"
Step 2 (Critique): "Wait, let me review this analysis...
  - Is X logically sound?
  - Are there edge cases I'm missing?
  - Does X match the ground truth?"
Step 3 (Refine): "Actually, I need to reconsider because..."
Step 4 (Output): "The correct answer is Y"
```

**Reflexion pattern:** Agent receives feedback, reflects on failures, updates strategy

**Critical for agents:** Enables learning from mistakes and iterative improvement

---

## Temperature Parameter

**Definition:** Controls randomness or creativity of generated text.

**How it works:**

- Higher temperature (e.g., 0.8): More diverse and creative outputs
- Lower temperature (e.g., 0.2): More focused and conservative responses

**Why it matters:** Temperature affects the sampling process during text generation

- Higher temperatures encourage exploring less likely word choices
- Lower temperatures favor more probable and safe options

**For agents:**

- Tool selection: Use low temperature (deterministic choices)
- Creative generation: Use higher temperature
- Reasoning tasks: Use moderate temperature (balance creativity and consistency)

---

## Prompt Tuning and P-Tuning (PEFT (Parameter-Efficient Fine-Tuning) Methods)

Both techniques belong to **prompt learning methods** - they customize LLMs by inserting **virtual tokens** (soft tokens) into the model input.

### Soft Tokens

Soft tokens are learnable embedding vectors that exist only in the model's continuous embedding space - they have **no corresponding word or symbol** in the vocabulary.

**Why they're powerful:** Soft tokens can encode task-specific information that may not be expressible with natural language prompts. They operate in the same embedding space the model uses, allowing them to influence model behavior more precisely than hand-crafted text prompts.

Think of soft tokens like **tuning knobs** you add to the model's input:

- Hard tokens = specific words you say to the model
- Soft tokens = learned "settings" that adjust how the model interprets and responds to your words

### Prompt Tuning

**Mechanism:** Soft prompt embeddings initialized as a 2D matrix of size `total_virtual_tokens × hidden_size`. Each task has its own independent embedding matrix.

**Characteristics:**

- Tasks do NOT share parameters during training or inference
- Each task maintains its own independent embedding table
- Based on the paper: "The Power of Scale for Parameter-Efficient Prompt Tuning"

### P-Tuning

**Mechanism:** Uses an intermediate model (LSTM or MLP) called a **prompt encoder** to _predict_ virtual token embeddings, rather than learning them directly.

**Characteristics:**

- Prompt encoder parameters randomly initialized
- All base LLM parameters frozen - only prompt encoder weights updated
- After training, virtual tokens move to a prompt table and encoder removed
- Enables continuous multitask learning without catastrophic forgetting
- Based on the paper: "GPT Understands, Too"

### Why These Techniques Matter

| Aspect                  | Benefit                                                   |
| ----------------------- | --------------------------------------------------------- |
| **Resource efficiency** | Tunes <0.1% of model parameters                           |
| **Speed**               | Training can complete in ~20 minutes                      |
| **Memory**              | No need to store full fine-tuned model copies             |
| **Preservation**        | Original LLM knowledge remains intact (frozen weights)    |
| **Multitasking**        | Same model can handle multiple p-tuned/prompt-tuned tasks |

### NVIDIA NeMo PEFT Methods

**NeMo 2.0 (Current):**

- LoRA - Low-rank decomposition matrices
- DoRA - Magnitude + direction decomposition with LoRA

**NeMo 1.0 (Legacy):**

- P-Tuning - Prompt encoder generates virtual tokens
- Prompt Tuning - Direct soft token learning
- LoRA and QLoRA - Low-rank adaptation variants
- Adapters - Bottleneck layers in transformer
- IA3 - Rescaling vectors (fewest parameters)

---

## NVIDIA NIM Prompt Templates

### Overview

Prompt templates in NVIDIA NIMs use **Jinja2 templating syntax** to structure inputs for LLM inference. Templates follow the **OpenAI/NIM format** for chat completion with standardized message roles.

Used across: Evaluation, tool calling, guardrails, synthetic data generation, reasoning control

### Jinja2 Template Syntax

**Variables:** `{{variable}}` or `{{item.property}}`

**Filters:** `{{text | trim}}`, `{{data | tojson}}`, `{{num | round(2)}}`

**Conditionals:** `{% if condition %} ... {% else %} ... {% endif %}`

**Loops:** `{% for item in items %} ... {% endfor %}`

### OpenAI/NIM Message Format

Templates structure conversations using standardized roles:

- `system`: Sets overall behavior, constraints, persona
- `user`: User queries/inputs
- `assistant`: Model responses
- `tool`: Results from external tool calls

### Use Cases in NVIDIA NIMs

1. **Evaluation Templates** - Structure benchmark tasks with ground truth comparison
2. **Tool Calling Templates** - Define available tools/functions with descriptions and parameters
3. **Guardrail Templates** - Content safety and topic control instructions
4. **Reasoning Control Templates** - Enable/disable chain-of-thought via `/think` or `/no_think`
5. **Synthetic Data Generation** - NeMo Curator uses templates with placeholders for data generation

### Best Practices for NIM Templates

- Keep templates modular (separate evaluation, tool calling, guardrail templates)
- Use clear variable names (`{{item.question}}` better than `{{x}}`)
- Include context in system prompts (define role, constraints, output format)
- Test with edge cases (empty inputs, multiple languages, long contexts)
- Validate output format (use filters like `| trim`, `| tojson`)
- For agents: Template should specify available tools, when to use them, expected formats

---

## Quick Reference: When to Use Each Technique

| Scenario                    | Technique                           | Why                                                     |
| --------------------------- | ----------------------------------- | ------------------------------------------------------- |
| Simple, well-known task     | Zero-shot                           | Fast, minimal setup                                     |
| Need specific format        | Few-shot                            | Examples show desired output format                     |
| Complex multi-step problem  | Chain-of-Thought                    | Explicit reasoning improves accuracy                    |
| Multi-tool agent            | ReAct + Tool Use                    | Interleaved reasoning and actions                       |
| Large task                  | Prompt Chaining                     | Break into smaller optimizable steps                    |
| Agent system                | System + Tool Use + Self-Reflection | Defines behavior, enables decisions, improves over time |
| Domain-specific             | Role Prompting + Few-shot           | Leverages domain expertise from training                |
| Quick model customization   | Prompt Tuning                       | <0.1% parameters, minimal training                      |
| Need knowledge preservation | P-Tuning                            | Frozen base, continuous multitask learning              |
