
import mechanize
import BeautifulSoup as bs4



br = mechanize.Browser()
br.addheaders =[('User-agent', 'Firefox')]


PMID = 0



for i in xrange(20000000, 21000000):

    PMID = i
    try:
        response = br.open("http://www.ncbi.nlm.nih.gov/pubmed/" + str(PMID))
        html = response.read()
        f= open("abstracts/" + str(PMID) + ".html", "w")
        f.write(html)
        f.close()
        print "processing pmid: " + str(PMID)

    except:
        print "couldn't open pmid: " + str(PMID)
