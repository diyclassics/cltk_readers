"""

"""


from typing import Union, Callable, Iterator

__author__ = ["Clément Besnier <clem@clementbesnier.fr>", ]


class CORAReader:
    def __init__(self, tree):
        self.tree = tree
        self.root = tree.getroot()

    @property
    def id(self):
        return self.tree.getroot().get("id", 0)

    @property
    def cora_headers(self):
        header_node = [child for child in self.root.getchildren() if child.tag == "header"]
        if header_node:
            header = header_node[0]
            return {n.tag: n.text for n in header.getchildren()}
        return None

    @property
    def layout_info(self):
        layout_info_node = [child for child in self.root.getchildren() if child.tag == "layoutinfo"]
        if layout_info_node:
            layout_info = layout_info_node[0]
            return [CORALayoutInfo.scan(n) for n in layout_info.getchildren()]
        return None

    @property
    def shift_tags(self):
        shift_tags_node = [child for child in self.root.getchildren() if child.tag == "shifttags"]
        if shift_tags_node:
            shift_tags = shift_tags_node[0]
            return [CORAShiftTags.scan(n) for n in shift_tags.getchildren()]
        return None

    @property
    def annotated_tokens(self):
        tokens = [CORAToken(child) for child in self.root.getchildren() if child.tag == "token"]
        return tokens

    def paras(self):
        """
        CorA documents do not have include paragraph divisions
        """
        raise NotImplementedError

    def sents(self, normalized=True):
        sentence_markers = ["."]
        # sentences = []
        current_sentence = []
        for word in self.words(normalized=normalized):
            current_sentence.append(word)
            if word in sentence_markers:
                yield current_sentence
                # sentences.append(current_sentence)
                current_sentence = []

    def words(self, normalized=True) -> Iterator[str]:
        for child in self.root.getchildren():
            if child.tag == "token":
                cora_token = CORAToken(child)
                if cora_token.tok_anno.pos == "$_":
                    yield cora_token.tok_anno.trans
                elif cora_token.tok_anno.norm:
                    if normalized:
                        yield cora_token.tok_anno.norm
                    elif cora_token.tok_anno.trans:
                        yield cora_token.tok_anno.utf
                else:
                    yield str(cora_token.tok_anno.to_dict())

    def lines(self):
        lines_layout = [i for i in self.layout_info if isinstance(i, CORALine)]
        d = {child.token_id: child for child in self.annotated_tokens}
        # lines = []

        for line in lines_layout:
            line_of_tokens = []
            for t_id in line.token_id_range:
                line_of_tokens.append(d[f"t{t_id}"])
            yield line_of_tokens
            # lines.append(line_of_tokens)
            # yield line_of_tokens


class CORAPage:
    TAG = "page"

    def __init__(self, page_node):
        self.page_node = page_node
        self.id = page_node.get("id")
        self.no = page_node.get("no")
        self.range = page_node.get("range")

    def __repr__(self):
        return f"page: {self.id} ({self.range})"

    def to_dict(self):
        return dict(id=self.id, no=self.no, range=self.range)


class CORAColumn:
    TAG = "column"

    def __init__(self, column_node):
        self.id = column_node.get("id")
        self.range = column_node.get("range").split("..")

    def __repr__(self):
        return f"column: {self.id} ({self.range})"

    def to_dict(self):
        return dict(id=self.id, range=self.range)


class CORALine:
    TAG = "line"

    def __init__(self, node):
        self.id = node.get("id")
        self.name = node.get("name")
        self.range = [int(i.split("_")[0].replace("t", "")) for i in node.get("range").split("..")]
        self.loc = node.get("loc").split(",")

    def __repr__(self):
        return f"line: {self.id} ({self.range})"

    def is_in_range(self, token_id):
        token_number = int(token_id.replace("t", ""))
        return self.range[0] <= token_number <= self.range[1]

    @property
    def token_id_range(self):
        return range(self.range[0], self.range[1]+1)

    def to_dict(self):
        return dict(id=self.id, name=self.name, range=self.range, loc=self.loc)


class CORALayoutInfo:

    @classmethod
    def scan(cls, node) -> Union[None, CORAPage, CORALine, CORAColumn]:
        if node.tag == CORAPage.TAG:
            return CORAPage(node)
        elif node.tag == CORALine.TAG:
            return CORALine(node)
        elif node.tag == CORAColumn.TAG:
            return CORAColumn(node)
        return None


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
        self.token_id = node.get("id")
        self.token_id_int = int(self.token_id.replace("t", ""))
        self.trans = node.get("trans")
        self.type = node.get("type")

        self.tok_dipl = node.find("tok_dipl")
        self.tok_anno = CORATokAnno(node.find("tok_anno"))

    def __repr__(self):
        return f"{self.trans}"


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
        for n in node.getchildren():
            if n.tag == "norm":
                self.norm = n.get("tag")
            if n.tag == "lemma":
                self.lemma = n.get("tag")
            if n.tag == "lemma_gen":
                self.lemma_gen = n.get("tag")
            if n.tag == "lemma_idmwb":
                self.lemma_idmwb = n.get("tag")
            if n.tag == "pos":
                self.pos = n.get("tag")
            if n.tag == "pos_gen":
                self.pos_gen = n.get("tag")
            if n.tag == "infl":
                self.infl = n.get("tag")
            if n.tag == "inflClass":
                self.inflClass = n.get("tag")
            if n.tag == "inflClass_gen":
                self.inflClass_gen = n.get("tag")
            if n.tag == "punc":
                self.punc = n.get("tag")

    def __repr__(self):
        if self.norm:
            return f"{self.norm}"
        else:
            return f"{self.trans}"

    def to_dict(self):
        return dict(id=self.id, trans=self.trans, utf=self.utf,
                    norm=self.norm, lemma=self.lemma, lemma_gen=self.lemma_gen,
                    lemma_idmwb=self.lemma_idmwb, pos=self.pos, pos_gen=self.pos_gen,
                    infl=self.infl, infl_class=self.inflClass, infl_class_gen=self.inflClass_gen,
                    punc=self.punc)




