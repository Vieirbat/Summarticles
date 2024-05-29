from bs4 import BeautifulSoup
from dateutil import parser
import re
##from chemtok import ChemTokeniser

import random

exp2 = re.compile(r"(\d)\s+(A[cglmrstu]|B[aehikr]?|C[adeflmnorsu]?|D[bsy]|E[rsu]|F[elmr]?|G[ade]|H[efgos]?|I[nr]?|Kr?|L[airuv]|M[dgnot]|N[abdeiop]?|Os?|P[abdmortu]?|R[abefghnu]|S[bcegimnr]?|T[abcehilm]|U(u[opst])?|V|W|Xe|Yb?|Z[nr])")

class Article:

    def __init__(self, stream):
        """
        Parses a TEI XML datastream and returns a processed article.

        Args:
            stream: handle to input data stream
        Returns:
            Article
        """
        soup = BeautifulSoup(stream, "lxml")
        self.exp1 = re.compile(r"(A[cglmrstu]|B[aehikr]?|C[adeflmnorsu]?|D[bsy]|E[rsu]|F[elmr]?|G[ade]|H[efgos]?|I[nr]?|Kr?|L[airuv]|M[dgnot]|N[abdeiop]?|Os?|P[abdmortu]?|R[abefghnu]|S[bcegimnr]?|T[abcehilm]|U(u[opst])?|V|W|Xe|Yb?|Z[nr])\s+(\d)")
        
        self.exp2 = re.compile(r"(\d)\s+(A[cglmrstu]|B[aehikr]?|C[adeflmnorsu]?|D[bsy]|E[rsu]|F[elmr]?|G[ade]|H[efgos]?|I[nr]?|Kr?|L[airuv]|M[dgnot]|N[abdeiop]?|Os?|P[abdmortu]?|R[abefghnu]|S[bcegimnr]?|T[abcehilm]|U(u[opst])?|V|W|Xe|Yb?|Z[nr])")


        self.title = soup.title.text
        self.published, self.publication, self.authors, self.reference = \
            self.parse_metadata(soup)
        self.abstract = self.parse_abstract(soup)
        self.text = self.parse_text(soup)
        self.design = random.choice(["Systematic Review","Randomized Trial", "Comparative Study","Descriptive Case"])



    def parse_metadata(self, soup):
        """
        Extracts article metadata.

        Args:
            soup: bs4 handle

        Returns:
            (published, publication, authors, reference)
        """

        # Build reference link
        source = soup.find("sourcedesc")
        if source:
            published = source.find("monogr").find("date")
            publication = source.find("monogr").find("title")

            # Parse publication information
            published = self.parse_date(published)
            publication = publication.text if publication else None
            authors = self.parse_authors(source)

            struct = soup.find("biblstruct")
            reference = "https://doi.org/" + struct.find("idno").text \
                        if struct and struct.find("idno") else None
        else:
            published, publication, authors, reference = None, None, None, None

        return (published, publication, authors, reference)

    def parse_authors(self, source):
        """
        Builds an authors string from a TEI sourceDesc tag.

        Args:
            source: sourceDesc tag handle

        Returns:
            list of authors
        """

        authors = []
        for name in source.find_all("persname"):
            surname = name.find("surname")
            forename = name.find("forename")

            if surname and forename:
                authors.append("%s, %s" % (surname.text, forename.text))

        return authors

    def parse_date(self, published):
        """
        Attempts to parse a publication date, if available.
        Otherwise, None is returned.

        Args:
            published: published object

        Returns:
            publication date if available/found, None otherwise
        """

        # Parse publication date
        # pylint: disable=W0702
        try:
            published = parser.parse(published["when"]) if published \
                        and "when" in published.attrs else None
        except Exception:
            published = None

        return published

    def parse_abstract(self, soup):
        """
        Find the abstract sections.

        Args:
            soup: bs4 handle

        Returns:
            abstract
        """

        abstract = soup.find("abstract").text
        if abstract:
            # Transform and clean text
            abstract = abstract.replace("\n", "")
            abstract = re.sub(self.exp1,r"\1\3",abstract)
            abstract = re.sub(self.exp2,r"\1\3",abstract)

        return abstract

    def parse_text(self, soup):
        """
        Builds a list of text sections.

        Args:
            soup: bs4 handle

        Returns:
            list of sections
        """

        sections = []

        for section in soup.find("text").find_all("div", recursive=False):
            # Section name and text
            children = list(section.children)

            # Attempt to parse section header
            if not children[0].name:
                name = str(children[0]).upper()
                children = children[1:]
            else:
                name = None

            text = " ".join([str(e.text) for e in children])
            text = text.replace("\n", "")
            text = re.sub(self.exp1,r"\1\3",text)
            text = re.sub(self.exp2,r"\1\3",text)


            sections.append((name, text))

        return sections

    def annotate_abstract(self, model, level = 0.01):
        
        tokens = model.process(self.abstract,level)
        tokens = {t[0]:t[2] for t in tokens}
        ct = ChemTokeniser(self.abstract,clm=True)
        self.annotated_text = []
        for t in ct.tokens:
            if t.start in tokens.keys():
                self.annotated_text.append((t.value,"chem","#fea"))
            else:
                self.annotated_text.append(t.value)
        return(self.annotated_text)
    
    def get_data(self):
        return({'title': self.title,
                'published' : self.published,
                'publication' : self.publication,
                'authors' : self.authors,
                'reference': self.reference,
                'abstract' : self.abstract,
                'text' : self.text,
                'design': self.design})
                #'annotated' : self.annotated_text})
