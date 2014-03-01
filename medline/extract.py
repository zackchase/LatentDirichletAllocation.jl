
from pyquery import PyQuery as pq
from lxml import etree
import urllib

import re
from nltk.corpus import stopwords

start_pmid = 20000000
end_pmid = 20008300

# read in each article
# extract abstract
# save

def extract():
    for i in xrange(start_pmid, end_pmid):

        index = str(i)

        print "processing document #:" + index


        try:
            f= open("abstracts/" + index +".html", "r")
            html = f.read()
            f.close()
            d = pq(html)
            abstract = d(".abstr")
            text = abstract.text()
            print text

            if len(text) > 500:
                g=open("abstract_text/" + index +".txt", "w")
                g.write(text)
                g.close()

        except:
            print "error on index: " + index


################################################################
#  Read in the abstracts and return array containing
###############################################################
def read_abstracts():


    abstracts = []
    for i in xrange(start_pmid, end_pmid):

        index = str(i)

        try:
            f = open("abstract_text/" + index +".txt", "r")
            html = f.read()
            abstracts.append(html)
            f.close()
            print "adding abstract: " + index
        except:
            print "abstract with pmid: " + index + " not found"

    return abstracts


def vocabulary(docs):
    vocabulary = []
    i=1
    for doc in docs:

        print "processing document: " + str(i)
        i += 1

        doc = doc.lower()
        doc = re.sub('[()!@#%\[\]&^*$.,\{\}\"\';:]', '', doc)

        for word in doc.split(" "):
            if word.count("-") >= 1:
                continue
            elif word not in vocabulary and word not in stopwords.words('english'):
                vocabulary.append(word)

    return vocabulary



