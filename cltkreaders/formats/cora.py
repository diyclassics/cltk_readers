"""

"""


__author__ = ["Clément Besnier <clem@clementbesnier.fr>", ]


class CORALayoutInfo:

    @classmethod
    def scan(cls, node):
        if node.tag == CORAPage.TAG:
            return CORAPage(node)
        elif node.tag == CORALine.TAG:
            return CORALine(node)
        elif node.tag == CORAColumn.TAG:
            return CORAColumn(node)
        return None


class CORAPage:
    TAG = "page"

    def __init__(self, page_node):
        self.page_node = page_node
        self.id = page_node.get("id")
        self.no = page_node.get("no")
        self.range = page_node.get("range")

    def to_dict(self):
        return dict(id=self.id, no=self.no, range=self.range)


class CORAColumn:
    TAG = "column"

    def __init__(self, column_node):
        self.id = column_node.get("id")
        self.range = column_node.get("range").split("..")

    def to_dict(self):
        return dict(id=self.id, range=self.range)


class CORALine:
    TAG = "line"

    def __init__(self, node):
        self.id = node.get("id")
        self.name = node.get("name")
        self.range = node.get("range").split("..")
        self.loc = node.get("loc").split(",")

    def to_dict(self):
        return dict(id=self.id, name=self.name, range=self.range, loc=self.loc)


class CORAShiftTags:
    TAG = "shifttags"

    @classmethod
    def scan(cls, node):
        if node.tag == CORAShiftTagsParen.TAG:
            return CORAShiftTagsParen(node)
        elif node.tag == CORAShiftTagsQuote.TAG:
            return CORAShiftTagsQuote(node)
        return None


class CORAShiftTagsParen:
    TAG = "paren"

    def __init__(self, node):
        self.range = node.get("range").split("..")


class CORAShiftTagsQuote:
    TAG = "quote"

    def __init__(self, node):
        self.range = node.get("range").split("..")


class CORAToken:
    TAG = "token"

    def __init__(self, node):
        self.tok_dipl = node.find("tok_dipl")
        self.tok_anno = CORATokAnno(node.find("tok_anno"))


class CORATokAnno:
    def __init__(self, node):
        self.node = node
        self.id = node.get("id")
        self.trans = node.get("trans")
        self.utf = node.get("trans")

        # children = node.children()
        # 'norm', 'lemma', 'lemma_gen', 'lemma_idmwb', 'pos', 'pos_gen', 'infl', 'inflClass', 'inflClass_gen', 'punc'
        self.norm = None
        self.lemma = None
        self.lemma_gen = None
        self.lemma_idmwb = None
        self.pos = None
        self.pos_gen = None
        self.infl = None
        self.inflClass = None
        self.inflClass_gen = None
        self.punc = None

        if node.find("norm"):
            self.norm = node.find("norm").get("tag")
        if node.find("lemma"):
            self.lemma = node.find("lemma").get("tag")
        if node.find("lemma_gen"):
            self.lemma_gen = node.find("lemma_gen").get("tag")
        if node.find("lemma_idmwb"):
            self.lemma_idmwb = node.find("lemma_idmwb").get("tag")
        if node.find("pos"):
            self.pos = node.find("pos").get("tag")
        if node.find("pos_gen"):
            self.pos_gen = node.find("pos_gen").get("tag")
        if node.find("infl"):
            self.infl = node.find("infl").get("tag")
        if node.find("inflClass"):
            self.inflClass = node.find("inflClass").get("tag")
        if node.find("inflClass_gen"):
            self.inflClass_gen = node.find("inflClass_gen").get("tag")
        if node.find("punc"):
            self.punc = node.find("punc").get("tag")

    def to_dict(self):
        return dict(id=self.id, trans=self.trans, utf=self.utf,
                    norm=self.norm, lemma=self.lemma, lemma_gen=self.lemma_gen,
                    lemma_idmwb=self.lemma_idmwb, pos=self.pos, pos_gen=self.pos_gen,
                    infl=self.infl, infl_class=self.inflClass, infl_class_gen=self.inflClass_gen,
                    punc=self.punc)




