"""
Seed documents for the GraphQ demo — auto-indexed on startup.
Public domain / educational texts covering diverse topics to
produce interesting co-occurrence graphs and meaningful search results.
"""

SEED_DOCUMENTS = [
    # Doc 0: Graph Theory
    (
        "Graph Theory Fundamentals",
        """Graph theory is the mathematical study of networks and their properties. A graph consists of vertices also called nodes and edges that connect pairs of vertices. Graphs can be directed where edges have a direction or undirected where connections are symmetric. Common applications include social network analysis where people are nodes and friendships are edges. Transportation networks model cities as vertices and roads as edges. The World Wide Web itself is a massive directed graph where web pages link to each other. Important concepts include degree which counts how many edges touch a vertex and path which is a sequence of edges connecting vertices. Graph algorithms solve problems like finding the shortest path between two nodes or detecting communities of densely connected vertices. Centrality measures identify the most important nodes in a network based on their position and connections."""
    ),
    # Doc 1: Machine Learning
    (
        "Introduction to Machine Learning",
        """Machine learning is a branch of artificial intelligence that enables computers to learn patterns from data without being explicitly programmed. Supervised learning uses labeled training examples to teach models how to map inputs to outputs. Common algorithms include linear regression for predicting continuous values and decision trees for classification tasks. Deep learning uses neural networks with many hidden layers to learn hierarchical representations of data. Each layer transforms the input into increasingly abstract features. Training a neural network involves forward propagation to compute predictions and backpropagation to update weights using gradient descent. Unsupervised learning finds hidden structure in unlabeled data through clustering and dimensionality reduction. Reinforcement learning trains agents to make sequences of decisions by rewarding desired behaviors and penalizing mistakes. Applications range from image recognition and natural language processing to recommendation systems and autonomous vehicles."""
    ),
    # Doc 2: Information Retrieval
    (
        "How Search Engines Work",
        """Information retrieval systems help users find relevant documents from large collections. The fundamental challenge is matching a user query expressed as a few keywords against a corpus of millions of documents. Traditional approaches use the bag of words model where each document is represented as a vector of term frequencies. The TF-IDF weighting scheme gives more importance to rare terms that are discriminative and less weight to common words that appear everywhere. Modern search engines combine lexical matching with semantic understanding using neural network models. Vector space models represent both queries and documents as points in a high dimensional space where cosine similarity measures relevance. Inverted indexes enable fast lookup of which documents contain each term without scanning the entire collection. Ranking algorithms like PageRank use the link structure of the web as a signal of document authority and importance. Query expansion and relevance feedback help refine search results based on user interactions and clickthrough data."""
    ),
    # Doc 3: Natural Language Processing
    (
        "Natural Language Processing and Word Embeddings",
        """Natural language processing enables computers to understand analyze and generate human language. Early approaches relied on hand crafted rules and linguistic knowledge but modern NLP is dominated by statistical and neural methods. Word embeddings are dense vector representations where semantically similar words are close together in the vector space. The distributional hypothesis states that words appearing in similar contexts tend to have similar meanings. Word2Vec learns embeddings by predicting surrounding words from a target word using a shallow neural network. GloVe combines global matrix factorization with local context window methods to produce word vectors. These embeddings capture linguistic regularities such that the vector difference between king and queen is similar to the difference between man and woman. Transformer models like BERT and GPT use self attention mechanisms to process entire sequences in parallel and capture long range dependencies. Attention allows each word to directly attend to every other word in the input enabling the model to weigh the importance of different contextual cues."""
    ),
    # Doc 4: Network Science
    (
        "Network Science and Complex Systems",
        """Network science studies complex systems by representing them as graphs of interacting components. Many real world networks share surprising structural properties regardless of their domain. The small world phenomenon describes networks where most nodes can reach any other node through a small number of steps despite the network being large and sparse. Scale free networks have degree distributions that follow a power law meaning a few hub nodes have many connections while most nodes have very few. This pattern emerges in citation networks the internet social networks and biological interaction networks through preferential attachment where new nodes tend to connect to already well connected nodes. Community structure refers to groups of nodes that are more densely connected internally than with the rest of the network. Detecting communities reveals functional modules in biological networks social circles in friendship networks and topic clusters in collaboration networks. Resilience analysis studies how networks respond to failures and attacks with scale free networks being robust to random failures but vulnerable to targeted attacks on hubs."""
    ),
]
