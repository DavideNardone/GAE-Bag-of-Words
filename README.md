# GAE-Bag-of-Word for study affinity

**_GAE-Bag-of-Words_** (GAE-BoW) is a Natural Languange Processing (NLP) model developed during an hackathon project that took place in Salerno (Naples-Italy). The challenge concerned the development of an hybrid mobile application to lead young students in finding their training and professional paths, promoting innovative ways that link the academic and the business world. I and my colleagues developed in 24h a prototype mobile application, that based on a Machine Learning technique would’ve helped new students to decide their own study affinity area. In practice, we firstly invited the students to fill up a survey, in which we presented different questions regarding: topics of study, hobby, types of books, etc. Secondly, we adopted a NLP model (BoW) to process the input data (train set) and then we trained a ML model able to list several disciplines for which a studen is most prone to. The ML model is learned on the basis of about 1,000 surveys, submitted by students of different ares of study. <br>
We have won the first prize, which then has provided the complete development of the mobile application.

# Requirements

In order to fully use the module is need to satisfy the following requirements:

- Python 2.7 or greater <br>
- Scipy
- Numpy
- Ntk
- Scikit-learn
- Pandas
- Pillow==3.4.2
- (Facoltative): App Engine Flexible Environment
- (Facoltative): Flask==0.11.1
- (Facoltative): Gunicorn==19.6.0

# Usage

1. `git clone https://github.com/DavideNardone/GAE-Bag-of-Words.git` <br>

2. `unzip GAE-Bag-of-Words-master.py`

3. Run `python main.py` <br>

**NB**: In case you'd like just to use the ML model, you can skip the installation of the **_facoltative_** packages and simply run the **_learningModel.py_**.

# Authors

[Davide Nardone](https://www.linkedin.com/in/davide-nardone-127428102/), University of Naples Parthenope, Science and Techonlogies Departement, Msc Applied Computer Science.
  
[Francesco Battistone](https://www.linkedin.com/in/francesco-battistone-324308120/), University of Naples Parthenope, Science and Techonlogies Departement, Msc Applied Computer Science.
  
[Vincenzo SantoPietro](https://www.linkedin.com/in/vincenzosantopietro/), University of Naples Parthenope, Science and Techonlogies Departement, Msc Applied Computer Science

[Gianmaria Perillo](https://www.linkedin.com/in/gianmaria-perillo-b04679138/), University of Naples Parthenope, Science and Techonlogies Departement, Msc Applied Computer Science

[Antonio Liguori](https://www.researchgate.net/profile/Antonio_Liguori2), University of Naples Parthenope, Science and Techonlogies Departement, Msc Applied Computer Science

For any kind of problem, questions, ideas or suggestions, please don't esitate to contact us.
