# Graph neural nets and Learning non-euclidean Spaces 


### Survey Papers 

- [A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/abs/1901.00596)
	<details>
	  <summary>Authors and Abstract</summary>

		by Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, Philip S. Yu

	   <p> 
		Deep learning has revolutionized many machine learning tasks in recent years, ranging from image classification and video processing to speech recognition and natural language understanding. The data in these tasks are typically represented in the Euclidean space. However, there is an increasing number of applications where data are generated from non-Euclidean domains and are represented as graphs with complex relationships and interdependency between objects. The complexity of graph data has imposed significant challenges on existing machine learning algorithms. Recently, many studies on extending deep learning approaches for graph data have emerged. In this survey, we provide a comprehensive overview of graph neural networks (GNNs) in data mining and machine learning fields. We propose a new taxonomy to divide the state-of-the-art graph neural networks into different categories. With a focus on graph convolutional networks, we review alternative architectures that have recently been developed; these learning paradigms include graph attention networks, graph autoencoders, graph generative networks, and graph spatial-temporal networks. We further discuss the applications of graph neural networks across various domains and summarize the open source codes and benchmarks of the existing algorithms on different learning tasks. Finally, we propose potential research directions in this fast-growing field.
	  </p>  
	</details>


- [Deep Learning on Graphs: A Survey (December 2018) Pre=print WIP](https://arxiv.org/abs/1812.04202) 
	by Ziwei Zhang, Peng Cui, Wenwu Zhu
- [Graph Neural Networks: A Review of Methods and Applications March 2019 ](https://arxiv.org/pdf/1812.08434.pdf) 
	 by Zhou et al 

- [Representation Learning on Graphs: Methods and Applications (2017)](https://cs.stanford.edu/people/jure/pubs/graphrepresentation-ieee17.pdf)
	<details>
	  <summary>Authors and Abstract</summary>

		by William Hamilton, Rex Ying and Jure Leskovec
	  <p>
		Machine learning on graphs is an important and ubiquitous task with applications ranging from drug design to friendship recommendation in social networks. The primary challenge in this domain is finding a way to represent, or encode, graph structure so that it can be easily exploited by machine learning models. Traditionally, machine learning approaches relied on user-defined heuristics to extract features encoding structural information about a graph (e.g., degree statistics or kernel functions). However, recent years have seen a surge in approaches that automatically learn to encode graph structure into low-dimensional embeddings, using techniques based on deep learning and nonlinear dimensionality reduction. Here we provide a conceptual review of key advancements in this area of representation learning on graphs, including matrix factorization-based methods, random-walk based algorithms, and graph convolutional networks. We review methods to embed individual nodes as well as approaches to embed entire (sub)graphs. In doing so, we develop a unified framework to describe these recent approaches, and we highlight a number of important applications and directions for future work.
	  </p>
	</details>




### Other Papers

#### Language Models 

- [DeepWalk: Online Learning of Social Representations (2014)](https://arxiv.org/pdf/1403.6652.pdf)
	<details>
	  <summary>Authors and Abstract</summary>

		by Bryan Perozzi, Rami Al-Rfou and Steven Skiena

	  <p>
		We present DeepWalk, a novel approach for learning latent representations of vertices in a network. These latent representations encode social relations in a continuous vector space, which is easily exploited by statistical models. DeepWalk generalizes recent advancements in language modeling and unsupervised feature learning (or deep learning) from sequences of words to graphs. DeepWalk uses local information obtained from truncated random walks to learn latent representations by treating walks as the equivalent of sentences. We demonstrate DeepWalk’s latent representations on several multi-label network classification tasks for social networks such as BlogCatalog, Flickr, and YouTube. Our results show that DeepWalk outperforms challenging baselines which are allowed a global view of the network, especially in the presence of missing information. DeepWalk’s representations can provide F1 scores up to 10% higher than competing methods when labeled data is sparse. In some experiments, DeepWalk’s representations are able to outperform all baseline methods while using 60% less training data. DeepWalk is also scalable. It is an online learning algorithm which builds useful incremental results, and is trivially parallelizable. These qualities make it suitable for a broad class of real world applications such as network classification, and anomaly detection.
	  </p>
	</details>

#### Social Networks 

- [Supervised Community Detection with Line Graph Neural Networks (2018)](https://arxiv.org/pdf/1705.08415.pdf)
	<details>
	  <summary>Authors and Abstract</summary>

		by Zhengdao Chen, Lisha Li3, and Joan Bruna

	  <p>
		We study data-driven methods for community detection on graphs, an inverse problem that is typically
		solved in terms of the spectrum of certain operators or via posterior inference under certain probabilistic
		graphical models. Focusing on random graph families such as the stochastic block model, recent research
		has unified both approaches and identified both statistical and computational signal-to-noise detection
		thresholds.
	  </p>
	</details>

- [Learning multi-faceted representations of individuals from heterogeneous evidence using neural networks (2015)](https://arxiv.org/abs/1510.05198)
	<details>
	  <summary>Authors and Abstract</summary>

		by Jiwei Li, Alan Ritter and Dan Jurafsky
	  <p>
		Inferring latent attributes of people online is an important social computing task, but requires integrating the many heterogeneous sources of information available on the web. We propose learning individual representations of people using neural nets to integrate rich linguistic and network evidence gathered from social media. The algorithm is able to combine diverse cues, such as the text a person writes, their attributes (e.g. gender, employer, education, location) and social relations to other people. We show that by integrating both textual and network evidence, these representations offer improved performance at four important tasks in social media inference on Twitter: predicting (1) gender, (2) occupation, (3) location, and (4) friendships for users. Our approach scales to large datasets and the learned representations can be used as general features in and have the potential to benefit a large number of downstream tasks including link prediction, community detection, or probabilistic reasoning over social networks.
	  </p>
	</details>

#### Representation Based - Graph Structure  

- [node2vec: Scalable Feature Learning for Networks (Stanford, 2016)](https://arxiv.org/abs/1607.00653) 
	<details>
	  <summary>Authors and Abstract</summary>

		by Aditya Grover and Jure Leskovec
	  <p>
		Prediction tasks over nodes and edges in networks require careful effort in engineering features used by learning algorithms. Recent research in the broader field of representation learning has led to significant progress in automating prediction by learning the features themselves. However, present feature learning approaches are not expressive enough to capture the diversity of connectivity patterns observed in networks. Here we propose node2vec, an algorithmic framework for learning continuous feature representations for nodes in networks. In node2vec, we learn a mapping of nodes to a low-dimensional space of features that maximizes the likelihood of preserving network neighborhoods of nodes. We define a flexible notion of a node’s network neighborhood and design a biased random walk procedure, which efficiently explores diverse neighborhoods. Our algorithm generalizes prior work which is based on rigid notions of network neighborhoods, and we argue that the added flexibility in exploring neighborhoods is the key to learning richer representations. We demonstrate the efficacy of node2vec over existing state-of-the-art techniques on multi-label classification and link prediction in several real-world networks from diverse domains. Taken together, our work represents a new way for efficiently learning state-of-the-art task-independent representations in complex networks.
	  </p>
	</details>


- [Deep Feature Learning for Graphs](https://arxiv.org/pdf/1704.08829.pdf)
	<details>
	  <summary>Authors and Abstract</summary>

		by Ryan A. Rossi, Rong Zhou, Nesreen K. Ahmed
	  <p>
		This paper presents a general graph representation learning framework called DeepGL for learning deep node and edge representations from large (attributed) graphs. In particular, DeepGL begins by deriving a set of base features (e.g., graphlet features) and automatically learns a multi-layered hierarchical graph representation where each successive layer leverages the output from the previous layer to learn features of a higher-order.
	  </p>
	</details>

 
- [Deep Neural Networks for Learning Graph Representations (2016)](https://pdfs.semanticscholar.org/1a37/f07606d60df365d74752857e8ce909f700b3.pdf) 
	<details>
	  <summary>Authors and Abstract</summary>

		by Shaosheng Cao, Wei Lu and Qiongkai Xu
	  <p>
		In this paper, we propose a novel model for learning graph representations, which generates a low-dimensional vector representation for each vertex by capturing the graph structural information. Different from other previous research efforts, we adopt a random surfing model to capture graph structural information directly, instead of using the samplingbased method for generating linear sequences proposed by Perozzi et al. (2014). The advantages of our approach will be illustrated from both theorical and empirical perspectives. We also give a new perspective for the matrix factorization method proposed by Levy and Goldberg (2014), in which the pointwise mutual information (PMI) matrix is considered as an analytical solution to the objective function of the skipgram model with negative sampling proposed by Mikolov et al. (2013). Unlike their approach which involves the use of the SVD for finding the low-dimensitonal projections from the PMI matrix, however, the stacked denoising autoencoder is introduced in our model to extract complex features and model non-linearities. To demonstrate the effectiveness of our model, we conduct experiments on clustering and visualization tasks, employing the learned vertex representations as features. Empirical results on datasets of varying sizes show that our model outperforms other state-of-the-art models in such tasks.
	  </p>
	</details>

- [Graph Matching Networks for Learning the Similarity of Graph Structured Objects 2019 ICML](https://arxiv.org/pdf/1904.12787.pdf)
	<details>
	  <summary>Authors and Abstract</summary>
	
		DeepMind: Yujia Li, Chenjie Gu, Thomas Dullien, Oriol Vinyals, Pushmeet Kohli 
	  <p>
		This paper addresses the challenging problem of retrieval and matching of graph structured objects, and makes two key contributions. First, we demonstrate how Graph Neural Networks (GNN), which have emerged as an effective model for various supervised prediction problems defined on structured data, can be trained to produce embedding of graphs in vector spaces that enables efficient similarity reasoning. Second, we propose a novel Graph Matching Network model that, given a pair of graphs as input, computes a similarity score between them by jointly reasoning on the pair through a new cross-graph attention-based matching mechanism. We demonstrate the effectiveness of our models on different domains including the challenging problem of control-flow-graph based function similarity search that plays an important role in the detection of vulnerabilities in software systems. The experimental analysis demonstrates that our models are not only able to exploit structure in the context of similarity learning but they can also outperform domain-specific baseline systems that have been carefully hand-engineered for these problems.
	  </p>
	</details>

#### Gated / Recurrence Related  
- [Gated Graph Sequence Neural Networks (Toronto and Microsoft, 2017) - ICLR](https://arxiv.org/pdf/1511.05493.pdf) 

	<details>
	  <summary>Authors and Abstract</summary>

		by Yujia Li, Daniel Tarlow, Marc Brockschmidt and Richard Zemel

	  <p>
		Graph-structured data appears frequently in domains including chemistry, natural language semantics, social networks, and knowledge bases. In this work, we study feature learning techniques for graph-structured inputs. Our starting point is previous work on Graph Neural Networks (Scarselli et al., 2009), which we modify to use gated recurrent units and modern optimization techniques and then extend to output sequences. The result is a flexible and broadly useful class of neural network models that has favorable inductive biases relative to purely sequence-based models (e.g., LSTMs) when the problem is graph-structured. We demonstrate the capabilities on some simple AI (bAbI) and graph algorithm learning tasks. We then show it achieves state-of-the-art performance on a problem from program verification, in which subgraphs need to be matched to abstract data structures.
	  </p>
	</details>

#### Convolution Related 
[Graph Classification with 2D Convolutional Neural Networks - Feb 2018](https://arxiv.org/pdf/1708.02218.pdf) 
	<details>
		<summary>Abstract</summary>
			<p>
		Graph learning is currently dominated by graph kernels, which, while powerful, suffer some significant limitations. Convolutional Neural Networks (CNNs) offer a very appealing alternative, but processing graphs with CNNs is not trivial. To address this challenge, many sophisticated extensions of CNNs have recently been introduced. In this paper, we reverse the problem: rather than proposing yet another graph CNN model, we introduce a novel way to represent graphs as multi-channel image-like structures that allows them to be handled by vanilla 2D CNNs. Experiments reveal that our method is more accurate than state-of-the-art graph kernels and graph CNNs on 4 out of 6 real-world datasets (with and without continuous node attributes), and close elsewhere. Our approach is also preferable to graph kernels in terms of time complexity. Code and data are publicly available.
	</p>
</details>


### Blog 
- [Viewing Matrices & Probability as Graphs](https://www.math3ma.com/blog/matrices-probability-graphs)
- [Graph Convolutional Networks, by Kipf (2016)](http://tkipf.github.io/graph-convolutional-networks/)

### Slides  
[A Short Tutorial on Graph Laplacians, Laplacian Embedding, and Spectral Clustering](https://csustan.csustan.edu/~tom/Clustering/GraphLaplacian-tutorial.pdf)
by Radu Horaud
