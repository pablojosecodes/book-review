# Why analyze models?
Key questions: What are our models learning / doing?
- Today’s models- black box
Questions we want to answer 
- What can we (not) learn via pretraining?
- How do our models affect people?
- What will replace the transformer?
- What do neural models tell us about language?

Varying levels of abstraction,from which to reaso about model
1. Probability distirbutino + decision function
2. Sequence of vector representation
3. Weights, mechanisms like attention, dropout, +++

# Out of domain Evaluation Sets

## Evaluation as analysis
- Concerned with **behavior**, not **mechanisms**- how does model behave in specific situation

Model trained on samples $\sim D$, how does it behave on samples from same $D$?
- ie. i.i.d.
- This is **test set accuracy / F1 / BLEU**

In Natural Language Inference task (recall NLI)
- Use specific carefully built diagnostic test set
	- E.g. HANS- tests syntactic heuristics in NLI

Humans as linguistic test subjects
- How do we undersand language behavior in humans? 
	- Minimal pairs: small change whicih makes unacceptable
- Just like in HANS- test set with careful properties
	- Question can LMs handle “attractors” in subject-verb agreement?

Careful test sets are kind of like unit test suites
- Minimum functionality tet sts for a specific behavior

# Inlfuence Studies + Adversarial Examples

## Input influence

Example: does my model really use long-distance context?
- Test: shuffle/rmoeve context farther than k words away- check loss

**Saliency map**: Prediction explanation for a single input
How to make?
- Lots of ways to encode “importance”

Simple gradient methods
- For words and model’s score, take the norm of the gradient of score wrt each word
	- $\text{salience}(x_i) = ||\nabla_{x_i}s_c(x_1,\ldots,x_n)||$ 
	- Idea: High gradient norm- changing word locally would affect score a lot
- Not perfect! Lots more methods
	- Example issue: linear approx may not hold well


**Explanation by Input reduction**
Idea: input saliency + then remove most unimportnat words
- can we break with seemingly innocuous changes to inut?


# Analyzing Interpretations
Idea: some modeling components lend to inspection
- Ie some heads seem to have simple operations, etc.

**Probing**: supervised analysis of nueral neural
- Given property y (like POS)
- Given model’s representations at a fixed layer
- Given function family
- Freeze parametrs of model, then train probe
	- $\hat y \sim f (h _i)$ where  $f \in F$ 
	- Extent to which can predict $y$ from $h_i$ shows accessibilty of feature in representation

Layerwise trends of probing accuracy
- More abstract linguistic properties are more accessible later in the network

*Summary* of probing / correlation studies
- Probing show that properties are accesible to probe fmail, not that thye’r eused by neural model you’re studying
- Likewise for correlation studies
- For example:
	- Hewitt and Liang, 2019 show that under certain conditions, probes can achieve high accuracy on random labels.
	- Ravichander et al., 2021 show that probes can achieve high accuracy on a property even when the model is trained to know the property isn’t useful.
- Some efforts towards causal studies- harder but interesting


# Model Ablation

Model tweaks + ablation as anlaysis
Typical NN improvement process is kind of a model analysis
- Tweaking, see if can remove complex parts, +++

