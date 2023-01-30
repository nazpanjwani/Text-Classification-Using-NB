import ast
from sklearn import metrics
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox
import plotly.figure_factory as ff
root=tk.Tk()
root.withdraw()

def NB_CLassifier():
    f= open("top_noun.txt", 'r')
    tmp=f.read()
    top_noun = ast.literal_eval(tmp)
    f.close()

    f= open("top_LF.txt", 'r')
    tmp=f.read()
    top_LF = ast.literal_eval(tmp)
    f.close()

    f= open("DataFrame.txt", 'r')
    tmp=f.read()
    df = ast.literal_eval(tmp)
    f.close()

    final_list = list(set().union(top_LF, top_noun))
    f = open("final_list.txt", "w")
    f.write(str(final_list))
    f.close()

    #Calculating Tf-idf of top 100 features
    X=df['Data']
    Y=df['Type']
    vectorizer = TfidfVectorizer(tokenizer = word_tokenize , ngram_range=(1,2) , binary=True , max_features=100)
    vectorizer.fit(final_list)
    X_tfidf = vectorizer.transform(X)

    #Applying Naive Bayes
    X_train, X_test , Y_train , Y_test = train_test_split(X_tfidf, Y, test_size = 0.3, random_state=3 , stratify = Y)

    model = MultinomialNB()
    model.fit(X_train.todense(), Y_train)
    Y_Pred = model.predict(X_test.todense())
    print('\nAccuracy score with Multinomial Naive Bayes is : ',metrics.accuracy_score(Y_test,Y_Pred)*100)
    print()
    matrix = confusion_matrix(Y_test, Y_Pred)
    print(classification_report(Y_test, Y_Pred))

    #Visualizing Confusion Matrix
    label = ["Course" , "No Course"]
    label_text = [[str(x) for x in label] for label in matrix]
    fig=ff.create_annotated_heatmap(matrix, x=label, y=label, annotation_text=label_text, colorscale='Viridis')

    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>')

    fig.add_annotation(dict(font=dict(color="black",size=14),
    x=0.5,
    y=-0.15,
    showarrow=False,
    text="Predicted value",
    xref="paper",
    yref="paper"))

    fig.add_annotation(dict(font=dict(color="black",size=14),
        x=-0.35,
        y=0.5,
        showarrow=False,
        text="Actual value",
        textangle=-90,
        xref="paper",
        yref="paper"))

    fig.update_layout(margin=dict(t=50, l=200))

    fig['data'][0]['showscale'] = True
    fig.show()

    ###Viewing classification Report
    messagebox.showinfo("Scores",classification_report(Y_test, Y_Pred))

NB_CLassifier()