---
title: Multimodal
---

# Early Models
Key question: How to align langauge and vision model?


Get some similarity metric and try to align the output vectors of 
- Skip gram language model
- Traditional visual model

Multimodal distributional semantics
- **Bag of visual words**: 
	- Take picture
	- Use algo (like SIFT) to find key points
	- Get feature descriptor for each key point
	- Cluster the feature descriptors with k means
	- Count how often feature descriptor occurs (ie. get the bag of words for the descriptors)
	- Concat visual word vector with text word vector 
	- Apply SVD to fuse information
- **Neural version**: applying deep learning to this idea
	- Concatenate features from convnets and word embeddings
	- Or- try to do skip gram prediction onto visual feature

Sentence level alignment
- 

Image to text: captioning

Attention

Text to image


# Features + Fusion

Features

Region features

Mutlimodal fusion

Early middle late

# Contrastive Models

CLIP

ALIGN




# Multimodal Foundation Models

VisualBERT

VilBERT

LXMERT

Supervised multimodal bitransformers

PixelBERT

UNITER

ViLT

Recommended paper: ___
- specifics don’t matter

FLAVA: holisitc- one fundational model- approach


CoCa

Frozen

Flamingo

Perceiver Resampler

Gated XATTEN

BLIP/BLIP2

Multimodal chain of thought

KSMOS-1


# Evaluation

COCO

VQA

CLEVR

Hateful memes

Winoground
# Beyond Images: other modalities
There are others
- Audio
- Video
- Olfactory embeddings
- Trimodal (audio, video, text)

Gounded language learning 
- Learning language by interacting in envirnoment? Someday in the future

Text to 3D


# Where to next?
- One foundation model will rule them all
	- Parameters will be shared in interesting ways
	- modality-agnostic foundation models- read + generate multi-modally
	- Automatic aligment from unpaired unimodal data will become a big topic
- Multimodal scaling laws
	- We will investigate tradeoffs
- Multi-Modal RAG
	- Query encoder: will be multimodal
	- Document index: will be multimodal
	- Generator: will be multimodal
- Beter evals + benchmarking