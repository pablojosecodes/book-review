---
title: Pretraining
---

# Subword Modelling

Byte-pair encoding
1. Start with vocab of only characters and EOW
2. Use ocrpus of text and find most common adjacent characters
3. Replace instances of character pari with new subword; repeat until desired vocab size
Vocab mapping
- e..g taaaaasty → (taa##, aaa###, sty)


# pretrained word embeddings → pretrained models

Better representations of language

# Pretraining
Reminder
- Encoders
	- Bidirectional context
- Encoder-Decoders
	- 
- Decoders
	- Langauge mdoels
	- Nice to generate from
# Encoder-only Pretraining

Masked language modeling

BERT specifically
- Input subword →
	- MASK 80%
	- Random token 10%
	- Unchanged 10%

Limitations of pretrained encoders
- good for comprehension/representation + filling in text
- not great for autoregressive generation methods
# Encoder-Decoder Pretraining

Training objective?
- Span corruption: replace different-length spans with unique placeholders- decode out the spans that were removed
- Still looks like language modelling at the decoder side

Finetuning
- Can be finetuned for wide range of question types

# Decoder-only Pretraining
Pretrain as language model, then use as generator, finetuning their $p_\theta(w_t|w_{1:t-1})$

Finetuning: train a classifier on the last word’s hidden state
- $h_t, \ldots, h_T = \text{Decoder}(w_1, \ldots, w_T)$
- $y \sim Ah_T + b$





# Section 2: **Prompting + RLHF**

# Zero Shot + Few Shot In Context Learning

GPT has emergent capabilities
GPT review
- Transformer decoder- 12 layers
GPT-2 (same architecture but larger) showed **zero-shot learning**
- Learn task with no examples, no graident updates- just specifying prediction problem
- beat SoTa on langauge modeling without task-specific fine-tuning

GPT-3: few shot learning (in context learning)
- Specify task by prepending examples of task before examples
- No gradient updates

Chain of thought prompting (also emergent property of model scale)
- Demonstrate sample chain of thought

Zero-shot chain of thought prompting
- “Let’s think step by step”
- 

# Instruction Finetuning
Collect examples (instruction / output pairs) + finetune language model
Then, evaluate on unseen tasks

As per usualy- data/mode scale is required

New benchmarks for multitask LMs
- MMLU- diverse knowledge intensive tasks
- BIGBench-200+ tasks like ASCII art meaning 

Limitations of finetuning
- Ground truth data = expensive
- Open ended creative generation have no right answer
- LMs penalie token level mistakes equally- but some erros are worse than other (ie. comparing blue vs tasty or green)=

# RLHF
Goal: somehow train language model to optimize for human preferences

RL
- Typical policy gradient stuff, trying to learn a policy $\pi_\theta$ to optimize expected human preferences $E[R]$ 
Issue: we need a non-differentiable reward function $R(s)$ that models human preferences
 Solution: model preferences with separate NLP problem
- Train Language model $RM_\phi(s)$ to predict human preferences, and then optimze for $RM_\phi$
Issue 2: human judgements are miscalibrated
Solution: ask for pairwise comparisons, then use a paried comparison model
- $J_{RM}(\phi) = -E_{(s^w, s^l) \sim D}[\text{log} \sigma(\text{RM}_\phi(s^w)-\text{RM}_\phi(s^l))]$ 
	- $s^w$: winning sample
	- $s^l$: losing sample

Make sure reward model works first!

Now we have
- Pretrained (possibly instruction finetuned) LM $p^{PT}(s)$
- Reward model $RM_\phi(s)$ which produces scalar rewards for LM outputs
- Method for optimizing LM parametesr towarsd reward function
To do RLHF
- Initialize copy of model $p_\theta^{RL}(s)$ with parameters $\theta$ that we would like to optimize
- Optimize the following reward with RL
	- $R(s) = RM_\phi(s) - \beta\ \text{log} \frac{p_\theta^{RL}(s)}{p^{PT}(s)}$
	- $\beta \ldots$ terms is KL-divergence between the two terms

Limitations of RL + Reward Modeling
- Human preferences are unreeliable
- Models o fhuman preferences are even more unreliable

# What’s Next
RLHF is 
- Very fast-moving
- Still data expensive
- Likely not a panacea
RL from AI feedback?


