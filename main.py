import pandas as pd
import os
import pyLDAvis
df=pd.read_csv(r"data\rawDataDithDoDero.csv",index_col=[0])

def text2tokens(rawtext:str):
    tokens=rawtext.lower().split(" ")
    tokens=[token for token in tokens if len(token)>2]
    return tokens


    documments = [text2tokens(content) for content in df["sentence"]]
    if __name__ == '__main__':
        topic = Topic(cwd=os.getcwd())
        topic.create_dictionary(documents=documments)
        topic.create_corpus(documents=documments)
        topic.train_lda_model(n_topics=4)

Topic.show_topics()
data = pyLDAvis.gensim.prepare("output/model/lda.model", gensim_dtm, dictionary)




