---
title: Linguistics of Langauge Models
---

# Structure in human language

Underlying structure in language (rememebr dependency parsing)
- Dictates the rules of language

Implictly, we know complex rules

**Grammar**: attempt to descrbe all these rules

Grammaticality: whether or not we consider an utterance in accordance with the grammar
- Some grammaticality rules accept useless utterances
- And block communicative utterances,

So why have rules in the first place?
- Without, we’d have limitless expressive things


# Linguistic Structure in NLP

Before self supervised learning
- Goal was to reverse engineer + imitate human language system (Syntax + semantics + discourse)
- E.g. Parsing

Now, we don’t constrain our systems to know any syntax
- They just catch on to stuff!

Question:
- In human: syntactic structures exist indepdnetnly of words they have appeared with (e..g jabberwocky)
- True for langauge models?

Tested with COGS Benchmark: new word-structure combinations
- Task: semantic inerpretation
- Training / test sets have distinct words + structures in diff. roles

Can test a whole bunch of other stuff in language models
- How do they map syntactic structure to meaning?
- Does the latent space encode structural information?
- How do new words impact this?

# Going Beyond Pure Structure
Semantics matters a ton! Impacts the rules of language
- This is how we train language models! Embeddings

Meaning isn’t allways just individual words, though
- e.g. idioms, constructions
- Can test in langauge models (via acceptability)
# Multilinguality
Multilingual language models let us share parameters (high and low resource languages)

Key ideas
- **Language typology**- lots of diversity
	- Evidentiality
	- Morphemes per word
	- Describing motion
- **Language universals**- lots of similarities
	- Universal grammer in the chomskyan sense?
	- All deal with subject, object, modifiers, etc.

