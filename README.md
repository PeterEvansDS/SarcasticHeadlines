# Sarcastic Headlines

NLP Classification model built to determine whether a headline is sarcastic or not. 




- Constructed using Kaggle dataset, [News Headlines Dataset For Sarcasm Detection](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection), which sources the sarcastic headlines from The Onion, and the legitimate ones from The Huffington Post.

- This text data was analysed with an interactive [Tableau dashboard](https://public.tableau.com/views/HeadlineEDA/Dashboard1?:language=es-ES&:display_count=n&:origin=viz_share_link), using spaCy beforehand to extract the different parts of speech present in each set of headlines ([`prep.py`](https://github.com/PeterEvansDS/SarcasticHeadlines/blob/main/prep.py)). 

- Classification models were constructed using a bag of words breakdown of the headlines, then using only the parts of speech and entities extracted, and then both together ([`bow.py`](https://github.com/PeterEvansDS/SarcasticHeadlines/blob/main/bow.py)). The best performing model achieves a test accuracy of 82%.

![alt text](https://github.com/PeterEvansDS/SarcasticHeadlines/blob/main/images/dashboard.png)
