#  Construction of a research data policy similarity network

Master's student project at Maastricht University, Winter 2023

**Authors**: Jonas Yezhou Bei

**Supervisors**: Remzi Celebi, Rohan Nanda

This project uses [conventional commits](https://www.conventionalcommits.org) ([short version](https://d33wubrfki0l68.cloudfront.net/de5e032567b4cccae05bafd47636c4b20f84868d/61d9f/images/posts/2019-11-01-understanding-semantic-commit-messages-using-git-and-angular/commitizen-example.png)).
<br>
<br>

**Please refer to the [project kanban board](https://github.com/users/jonasybei/projects/1) for current progress.**

## Description

Data protection policies are more important than ever in today's era of big data. Within this field, research data policies play a special role as they are not only concerned with privacy issues, but also directly influence reproducibility of research and therefore credibility.

The aim of this project is to create an overview of the landscape of research data policy documents, showing documents that are correlated, how documents are correlated, as well as arguing which documents are most influential on others.

To achieve these goals, the project at hand aims to construct a document similarity network based on publicly available data policies stemming from Dutch universities and research institutes. The network construction is done solely by means of text analysis. Then, the network is analysed to find the most important documents and argue the reasoning behind their importance. Policy documents are gathered continuously as part of the research.

The project is split into two research questions:

# Research Question 1: What is the Document Similarity Network of the research data policies corpora?

To construct the document similarity network, syntactic and semantic similarity methods are applied to the documents. These methods allow measuring distances between documents, such that the n closest documents can be selected to be neighbours. The value of n is found experimentally such that the overlap to the citation is network is maximised. The methods used include embedding methods (such as TF-IDF, N- Grams, and machine learning embeddings) as well as distance measures (such as Cosine Distance and Word Mover's Distance). Additionally, dimensionality reduction will be experimentally applied to aid visualizing the network and lowering the computational cost of the task.

# Research Question 2: What are the most relevant policy documents of the network, and why?

Once the network has been constructed, network clusters are identified. For better explainability, the network is transformed into a knowledge graph by means of connecting similar documents using their respective topics. To extract topics, methods such as Named Entity Recognition or Topic Modelling (Latent Dirichlet Allocation) are considered. This allows to extract the most important documents and argue for the reasoning behind their importance.
