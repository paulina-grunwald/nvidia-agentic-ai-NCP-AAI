# NeMo Aligner and RLHF: Aligning Models to Human Preferences

## The RLHF Pipeline (Full Flow)

Here's the complete journey of taking a raw model and making it actually useful:

```
Base Model → SFT → Reward Model → PPO → Aligned Model
```

Each arrow represents a distinct training phase. Let's walk through what happens at each stage.

## Stage-by-Stage Breakdown

### Stage 1: Pre-training
You start with a base model trained on massive amounts of unlabeled text from the internet. It knows grammar, facts, and language patterns, but it's like a teenager who hasn't learned manners yet. It might generate toxic content, contradict itself, or ignore instructions.

### Stage 2: Supervised Fine-Tuning (SFT)
Here, you take your base model and fine-tune it on curated instruction-response pairs. Think of datasets like ShareGPT or Alpaca: thousands of examples of "here's a user question, here's a good response." This teaches the model to follow instructions and be more helpful.

**Exam trigger:** If a question mentions "fine-tuning on instruction-response pairs" or "curated examples of good outputs," you're in SFT territory.

### Stage 3: Reward Model
This is where things get interesting. You train a separate neural network to score responses. It learns from human preference data: "When given the same prompt, humans prefer response A over response B."

The reward model becomes a judge. It doesn't generate text; it scores how good a generated response is, based on patterns it learned from human feedback.

**Exam trigger:** "Train a model to score responses based on human preferences" = Reward Model. This is a different model from your main LLM.

### Stage 4: PPO (Proximal Policy Optimization)
PPO is a reinforcement learning algorithm. Now you use your original model as the policy and the reward model as the objective. You generate outputs, the reward model scores them, and PPO adjusts the model's weights to maximize those reward scores while staying close to the original SFT model (hence "proximal").

**Exam trigger:** "Use reinforcement learning to optimize a model based on reward signals" = PPO.

### Stage 5: Aligned Model
The final product. A model that follows instructions, respects human preferences, and avoids harmful outputs.

## DPO: A Simpler Alternative

Direct Preference Optimization (DPO) is the newer, simpler cousin of PPO. Here's the key difference:

- **PPO approach:** Train reward model → Use PPO to optimize against reward model
- **DPO approach:** Directly optimize from preference pairs, skip the reward model entirely

DPO learns the same preferences without needing that intermediate reward model training step. It's fewer moving parts, faster to iterate on, and often comparable in quality.

**Exam trigger:** "Simpler alternative to RLHF without training a separate reward model" = DPO.

## NeMo Aligner: The Specific Tool

NeMo Aligner is NVIDIA's component that implements RLHF alignment. It's the practical tool that executes the pipeline stages we just discussed. NeMo Aligner supports both PPO and DPO approaches.

**Critical exam distinction:** There are many NeMo components, and the exam loves confusing them:

- **NeMo Aligner:** Handles RLHF, PPO, DPO, alignment optimization
- **NeMo Curator:** Data preparation and preprocessing (NOT alignment)
- **NeMo Retriever:** Adds retrieval-augmented generation (NOT alignment)
- **NeMo Guardrails:** Safety and moderation controls (NOT alignment)
- **NeMo Framework:** The broader training framework that encompasses multiple tools

**Exam trigger:** "Which NeMo component handles alignment to human preferences?" = NeMo Aligner.

## When to Use What: Decision Matrix

Your model has a problem. Which solution do you pick?

| Problem | Root Cause | Solution |
|---|---|---|
| Model doesn't follow instructions well | Base model lacks instruction-following | Use SFT |
| Model generates harmful, toxic, or unhelpful content | Base model wasn't optimized for human preferences | Use RLHF (NeMo Aligner with PPO or DPO) |
| Want to align model but don't want reward model complexity | Want faster iteration with fewer components | Use DPO |
| Model lacks domain-specific knowledge | Training data didn't include that domain | Use fine-tuning on domain data or RAG |
| Model generates hallucinations in production | Model wasn't constrained during training | Use NeMo Guardrails for runtime filtering |

## Key Exam Insights

1. **PPO requires a reward model; DPO doesn't.** This is tested constantly.

2. **SFT comes before RLHF in the pipeline.** You need instruction-following before you can align to preferences.

3. **NeMo Aligner is the tool; RLHF is the methodology.** The exam might ask "which NeMo component" (answer: Aligner) or "which algorithm" (answer: PPO or DPO).

4. **Alignment is about human preferences, not safety.** That's why NeMo Guardrails is separate. Guardrails enforce hard rules; Aligner learns soft preferences.

---

## Exam-Style Practice Questions

**Question 1:** You're building a customer service chatbot. You fine-tuned a base model on instruction-response pairs, but users report that responses often miss the point or contain irrelevant information. You want to optimize the model to generate responses that humans actually prefer. Which component should you use?

A) NeMo Curator to preprocess customer feedback
B) NeMo Aligner with RLHF to optimize against human preferences
C) NeMo Retriever to add knowledge base search
D) NeMo Guardrails to filter responses

**Answer:** B) NeMo Aligner with RLHF. The problem is that the model isn't aligned with human preferences (not enough quality filtering like A or C would provide). You need RLHF to learn from human feedback about which responses are actually helpful. NeMo Guardrails (D) handles safety constraints, not preference alignment.

---

**Question 2:** Your team wants to align a model using human preference data, but you want to avoid the complexity of training a separate reward model. Which approach should you recommend?

A) PPO with a reward model
B) DPO without a reward model
C) SFT on preference pairs
D) NeMo Guardrails with custom rules

**Answer:** B) DPO without a reward model. DPO is explicitly designed to skip the reward model training step while still optimizing from preference pairs. PPO (A) requires the reward model. SFT (C) is earlier in the pipeline and doesn't use preference data directly. Guardrails (D) is for safety filtering, not preference alignment.

---

**Question 3:** You have a base language model and want to implement the full RLHF pipeline. In what order should you execute these steps: (1) Train reward model, (2) Supervised Fine-Tuning, (3) PPO optimization, (4) Pre-train base model?

A) 4 → 2 → 1 → 3
B) 4 → 1 → 2 → 3
C) 2 → 1 → 3 → 4
D) 1 → 2 → 4 → 3

**Answer:** A) 4 → 2 → 1 → 3. Pre-train first (4), then SFT on instruction pairs (2), train the reward model on human preferences (1), and finally use PPO to optimize the model against that reward signal (3). This is the canonical RLHF pipeline order.

---

**Question 4:** Your organization is evaluating two alignment strategies. Strategy A trains a separate reward model then uses PPO. Strategy B directly optimizes from preference pairs without a reward model. Which statement is true?

A) Strategy A (PPO) is always superior and should be preferred
B) Strategy B (DPO) is a simpler alternative that can achieve comparable results with fewer components
C) Strategy B can only work if you don't have preference pair data
D) Both strategies require NeMo Curator for data preprocessing

**Answer:** B) Strategy B (DPO) is a simpler alternative that can achieve comparable results with fewer components. Both can work well in practice; DPO just has fewer moving parts. DPO doesn't require a reward model (not always superior, not required A), doesn't depend on lack of data (C is backwards), and neither necessarily requires NeMo Curator for alignment specifically (D is wrong, though Curator might prep data earlier).
