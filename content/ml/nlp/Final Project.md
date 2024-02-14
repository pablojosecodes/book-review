
**Default research project**: MinBERT
Base implementation
1. Use base given code to implement MinBERT
2. Utilize pre-trained model weights + embeddings to perform sentiment analysis on two datasets
3. Train for sentiment classification
Extend: how to build robust embeddings which can perform well across a large range of different tasks, not just one
1. Adjust BERT embeddings to perform following 3 tasks
	1. Sentiment analysis
	2. Paraphrase detection
	3. Semantic textual similarity
2. Find relevant research paper for each improvement (some suggestions given)

Notes on finding research projects (how to build an economic model in your spare time)
1. Getting ideas
	1. Journals not great
	2. Look in media, news, etc. that aren’t about your topic area
	3. Conversations with people in business
2. Think through your own idea independently, somewhat thoroughly
3. Go find somebody else that did your idea but 10x better- ask yourself why you missed what they did
4. Give seminar
5. Planning

Winning default papers
1. Walk Less + Only Down Smooth Valleys
	1. Pretrained embedings from BERT for 3 fine-grained tasks"
	2. First- test ability to be tuend towards sentence sentiment classification only
	3. Then implement SMART which aims to tackle overffiting
	4. Apply multitask learning approaches that learn on all 3 aforementioned tasks
2. Basically
	1. INvestigate of pre-trained + fine-tuned BERT model on 3 downsteam prediction tasks when include
		1. Regularization (SMART)
		2. Multitask Learning with task-specific datasets
		3. Rich relational layers that xploit similarity between tasks
3. Approach
	1. Starting point: BERT
	2. Focusing on the 3 specific fine-tuning tasks (set up basic baselines)
	3. Extending
		1. Regularization of loss + optimizer step (SMART)
			1. Coded up themselves
		2. Round-robin multitask fine-tuning
			1. Baseline BERT assumes fine-tuning only on sentiment classification generalizes well to paragraphising and similarity prediction tasks- not true
			2. Instead, implement batch-level round-robin MTL routine (SST, paragraph, and STS ata)
			3. test 2 versions
		3. Rich relational layer combining similar tasks
			1. Adapt model to handle relations acros tasks
	4. Experiments
		1. 

Finding research topics
Generally 2 ways in science
1. Nails: improve ways to address specific domain problem of interest
2. Hammers: start with technical method + work out ways to extend / improve / etc.

Most projects
1. Find application / task + expore how to approach / solve it effectively, often with existing model
2. Implement complex neural arch. + demonstrate performance
	* Ideally find some way to tweak it- make it better (kind of #3)
3. Come up with new / vaiant neural network model + explore empirical success
4. Analysis project- anayze behavior of model- how it represents linguisitc knowledge or what kinds of phenomena it can handle / errors it makes
5. Rare theoretical project

Examples
- Using LSTM togenerate lyrics (adding components for metric structrus + rhyme)
- Complex neural model: implement differential neural computers + get to work (I believe building an implementatin of existing closed source paper)
- Got published- showed improvements to RNNLMs (Title: Improving Learning through Augmenting the Loss)
- Quantization of word vectors
	- Counted for class because evaluated on natural language tasks

Finding a place to start
1. Recent papers- 
	1. ACL Anthologgy
	2. Online proceedings of major ML conferences
		1. NeurIPS papers.nips.cc
		2. ICML
		3. ICLR
	3. Arxiv.org
2. Even better- look for interesting problem in the world!

If want to beat the state of the art, look at leadeerboards
- Paperswithcode
- nlpprogress

Modern Deep Learning NLP
- Most works- download big pre-trained model + work from there
- Recommended for practical projects- 
	- Transformer from Huggingface
	- Load a big pre-trained language model
	- Fine tune it for our task
	- Test it
Exciting areas now
- Evaluating / improving models for something other than accuraacy
- Empirical work on what PLMs have learned
- Transfer learning with ittle data
- Low resource stuff
- Addressing bias
- Scaling models down (pruning, quantization, etc.)
- More advanced fucntionality (compositionality, generalization, fast leraning (e.g. meta learning) on smaller problems)

Datasets
- catalog.ldc.upenn.edu
- Huggingface
- paperswithcode
- for machine translation: statmt.org
- for dependency parsing: universaldependencies.org

Example of doing research (e.g. orf applying NN to summariation)
1. Define task
	1. Summarization
2. Define dataset
	1. Search for academic datasets (already have baselines, helpful)
		1. e.g. newsroom summmarization dataset
	2. Or- define your own dataset 
		1. Fresh problem
		2. be creative
3. Dataset hygiene
	1. Separate test and dev test data splits
4. Define metric
	1. Search for well etablished metrics
	2. Summarization: ROUGE or human eval
5. Establish baselin
	1. Implement simple model first
	2. Summarization: LEAD-3 baseline
	3. Compute metrics on Train AND dev NOT test 
	4. often will have errors- analyze
6. Imlement existing neural net mdel
	1. Compute metric to train + dev
	2. Analyze output + error
7. Always be cose to the data (except final test set)
	1. Visualize dataset
	2. Collect statistics 
	3. Look at errors
	4. Analyze hyperparameters
8. Try out diferent models + variants (set up quick iteration)
	1. Fixed dwindow Nn
	2. RNN
	3. Recursive NN
	4. CNN
	5. Attention based Model
	6. Etc.
9. Ideally only use test set once.

Getting nns to train
- Be positive- they want to learn
- Takes time to get them fixed up