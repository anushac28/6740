# get belief and sentiment links from an annotation file
#    (and format them consistently)

from lxml import etree
import os
import warnings
import json
from file_ids import blog_ids, df_weird_ids, newswire_ids
from test_split import test_fnames


class _BaseBestFile(object):
    def __init__(self, doc_id, data_root, gold_ere=True):
        self.doc_id = doc_id
        self.data_root = data_root
        self.gold_ere = gold_ere

        src_fname = self._build_source_fname()
        ere_fname = self._build_ere_fname()
        predicted_ere_fname = self._build_predicted_ere_fname()
        #ann_fname = self._build_annotation_fname()

        with open(src_fname) as f:
        #with open(src_fname) as f:
            self.source = f.read()

        if self.gold_ere:
            self.ere = etree.parse(ere_fname).getroot()     #DOUBT 1 : WHAT IS ETREE
        else:
            self.ere = etree.parse(predicted_ere_fname).getroot()

        self.annotation = self._build_annotation_fname()
        #self.annotation = etree.parse(ann_fname).getroot()

        # create convenient dicts to find EREs by mention_id
        self.objects = {}
        self.mentions = {}
        for entity in self.ere.find('entities').findall('entity'):
            for mention in entity.findall('entity_mention'):
                self.objects[mention.attrib['id']] = entity
                self.mentions[mention.attrib['id']] = mention

        for relation in self.ere.find('relations').findall('relation'):
            for mention in relation.findall('relation_mention'):
                self.objects[mention.attrib['id']] = relation
                self.mentions[mention.attrib['id']] = mention

        for event in self.ere.find('hoppers').findall('hopper'):
            for mention in event.findall('event_mention'):
                self.objects[mention.attrib['id']] = event
                self.mentions[mention.attrib['id']] = mention

        fillers = self.ere.find('fillers')
        if fillers is not None:
            for filler in fillers.findall('filler'):
                self.mentions[filler.attrib['id']] = filler

        # get parsed data
        tok_json_fname, parsed_fname  = self._build_source_ann_fnames()

        try:
           
            #print tok_json_fname
            with open(tok_json_fname, "r") as f:
                #print f
                self.tokenized = json.load(f)

            with open(parsed_fname, "r") as f:

                # split into sentences
                conll = f.read().split('\n\n')[:-1]

                # split each sentence into tokens
                conll = [sent.splitlines() for sent in conll]

                # split each token into its attributes
                conll = [[tok.split('\t') for tok in sent] for sent in conll]

                self.conll = conll

        except IOError:
            warnings.warn('Parsed files not found.  Searching in e.g. {} & {}'
                          .format(tok_json_fname, parsed_fname))

    def _build_source_fname(self, dir="source"):
        raise NotImplementedError

    def _build_ere_fname(self):
        return os.path.join(self.data_root,
                            'ere',
                            '{}.rich_ere.xml'.format(self.doc_id))

    def _build_predicted_ere_fname(self):
        return os.path.join(self.data_root,
                            'predicted_ere',
                            '{}.predicted.map.rich_ere.xml'.format(self.doc_id))


    def _build_annotation_fname(self):
        '''return os.path.join(self.data_root,
                            'annotation',
                            '{}.best.xml'.format(self.doc_id))'''
        return etree.parse(os.path.join(self.data_root,
                            'annotation',
                            '{}.best.xml'.format(self.doc_id))).getroot()

    def _build_source_ann_fnames(self):
        base = self._build_source_fname(dir=os.path.join("source_ann",
                                                         "parsed"))
        return base + ".json", base + ".conll8.parsed"

    def offset_to_tokens(self, offset, length):
        """ Converts offset+length coordinates to sentence_id and token_id.

        Returns (sentence_start, token_start), (sentence_end, token_end)

        with the meaning that the first word in the span will be:

            self.conll[sentence_start][token_start]

        and the last word in the span will be:

            self.conll[sentence_end][token_end]

        See `parsed_span` to get the continuous sequence of tokens between.

        """

        first_sent_id = None
        first_tok_id = None
        #print offset
        #print length

        for sent_id, (sent_json, sent_conll) in enumerate(
                zip(self.tokenized['sentences'], self.conll)):
            #print first_sent_id

            for tok_id, (token_json, token_conll) in enumerate(
                    zip(sent_json['tokens'], sent_conll)):

                begin = token_json['characterOffsetBegin']
                end = token_json['characterOffsetEnd']

                if end < offset:
                    continue

                if (offset + length) <= begin:


                    if first_sent_id is not None:
                        #print "hello"
                        return ((first_sent_id, first_tok_id),
                                (sent_id, tok_id))
                    else:
                        return None, None

                if first_sent_id is None:
                    first_sent_id = sent_id
                    first_tok_id = tok_id

        return None, None

    def parsed_span(self, start, end, include_boundaries=False):
        """ Returns the continuous sequence of tokens between the bounds.

        Parameters
        ----------

            start, end: output of `offset_to_tokens`.
                Each is a tuple (sentence_id, token_id).

            include_boundaries: bool
                Whether to include end-of-sentence markers in the span.

        Returns
        -------

            span: list of tuples
                List of conll-shaped tuples.
                E.g. [(1, 'hi', _, 'UH', ...), (2, 'mom, _, 'NOUN', ...)]

        """
        start_sent, start_tok = start
        end_sent, end_tok = end

        if start_sent == end_sent:
            span = self.conll[start_sent][start_tok:end_tok]

        else:

            span = self.conll[start_sent][start_tok:]

            if include_boundaries:
                span.append([-1, 'EOS'])

            for sent in range(start_sent + 1, end_sent):
                span.extend(self.conll[sent])

                if include_boundaries:
                    span.append([-1, 'EOS'])

            span.extend(self.conll[end_sent][:end_tok])

        return span

    def get_post_id(self, offset):
        return ""

    def path_to_root(self, sent_id, tok_id, guard=0):

        if guard > 100:
            print("max depth")
            return []

        tok = self.conll[sent_id][tok_id]

        parent_id = int(tok[6])
        if parent_id != 0:
            return self.path_to_root(sent_id, parent_id - 1, guard + 1) + [
                self.conll[sent_id][tok_id]]
        else:
            return [self.conll[sent_id][tok_id]]


class _BaseForumFile(_BaseBestFile):

    data_type = 'forum'

    def get_post_id(self, offset):
        post_line_start = self.source.rfind("<post ", 0, offset)
        post_line_end = self.source.find("\n", post_line_start)
        post_line = self.source[post_line_start:post_line_end]
        assert "\n" not in post_line

        id_start = post_line.find('id="') + 4
        id_end = post_line.find('"', id_start)
        post_id = post_line[id_start:id_end]

        return post_id


class BestOldForumFile(_BaseForumFile):
    """Hexadecimal doc_id, old forum files"""

    def _build_source_fname(self, dir="source"):
        return os.path.join(self.data_root,
                            dir,
                            '{}.cmp.txt'.format(self.doc_id))


class BestEvalOldForumFile(_BaseForumFile):
    """Hexadecimal doc_id, old forum files"""

    def _build_source_fname(self, dir="df/source"):
        return os.path.join(self.data_root,
                            dir,
                            '{}.xml'.format(self.doc_id))

    def _build_ere_fname(self):
        return os.path.join(self.data_root,
                            'df/ere',
                            '{}.rich_ere.xml'.format(self.doc_id))

    def _build_predicted_ere_fname(self):
        return os.path.join(self.data_root,
                            'df/predicted_ere',
                            '{}.predicted.map.rich_ere.xml'.format(self.doc_id))

    def _build_source_ann_fnames(self):
        base = self._build_source_fname(dir=os.path.join("df",
                                                         "parsed"))
        return base + ".json", base + ".conll8.parsed"

    def _build_annotation_fname(self):
        return None


class BestNewForumFile(_BaseForumFile):
    """New-style forum files with multiple annotations per source file"""

    def _build_source_fname(self, dir="source"):
        src_id, _ = self.doc_id.rsplit("_", 1)
        return os.path.join(self.data_root,
                            dir,
                            '{}.xml'.format(src_id))

class BestNewEvalForumFile(_BaseForumFile):
    """New-style forum files with multiple annotations per source file"""

    def _build_source_fname(self, dir="df/source"):
        src_id, _ = self.doc_id.rsplit("_", 1)
        return os.path.join(self.data_root,
                            dir,
                            '{}.xml'.format(src_id))

    def _build_ere_fname(self):
        return os.path.join(self.data_root,
                            'df/ere',
                            '{}.rich_ere.xml'.format(self.doc_id))

    def _build_predicted_ere_fname(self):
        return os.path.join(self.data_root,
                            'df/predicted_ere',
                            '{}.predicted.map.rich_ere.xml'.format(self.doc_id))



class BestNewswireFile(_BaseBestFile):
    """Newswire data"""

    data_type = 'news'

    def _build_source_fname(self, dir="source"):
        return os.path.join(self.data_root,
                            dir,
                            '{}.xml'.format(self.doc_id))

class BestEvalNewswireFile(_BaseBestFile):
    """Newswire data"""

    data_type = 'news'

    def _build_source_fname(self, dir="nw/source"):
        return os.path.join(self.data_root,
                            dir,
                            '{}.xml'.format(self.doc_id))

    def _build_ere_fname(self):
        return os.path.join(self.data_root,
                            'nw/ere',
                            '{}.rich_ere.xml'.format(self.doc_id))

    def _build_predicted_ere_fname(self):
        return os.path.join(self.data_root,
                            'nw/predicted_ere',
                            '{}.predicted.map.rich_ere.xml'.format(self.doc_id))

    def _build_source_ann_fnames(self):
        base = self._build_source_fname(dir=os.path.join("nw",
                                                         "parsed"))
        return base + ".json", base + ".conll8.parsed"


    def _build_annotation_fname(self):
        return None


def iter_eval_best_files(data_root, gold_ere=True):
    from os import listdir
    from os.path import isfile, join

    blog_ids = [f[:-4] for f in listdir(data_root+"df/source/") ]
    newswire_ids = [f[:-4] for f in listdir(data_root+"nw/source/") ]

    for doc_id in blog_ids:
        yield BestEvalOldForumFile(doc_id, data_root, gold_ere=gold_ere)

    for doc_id in newswire_ids:
        yield BestEvalNewswireFile(doc_id, data_root, gold_ere=gold_ere)

    

def iter_best_files(data_root):
    for doc_id in blog_ids:
        yield BestOldForumFile(doc_id, data_root)

    for doc_id in df_weird_ids:
        yield BestNewForumFile(doc_id, data_root)

    for doc_id in newswire_ids:
        yield BestNewswireFile(doc_id, data_root)


def iter_best_split(data_root, split='test', gold_ere=True):

    if split == "eval" : 
        for doc in iter_eval_best_files(data_root, gold_ere=gold_ere):
            yield doc
    
    else : 
        for doc in iter_best_files(data_root):
            if split == 'test' and doc.doc_id in test_fnames:
                yield doc
            elif split == 'train' and doc.doc_id not in test_fnames:
                yield doc
    

def extract_belief(annotations):
    for trg in annotations:
        links = trg.find('beliefs').getchildren()
        for link in links:
            res = {'trg': trg.attrib['ere_id'],
                   'belief_type': link.attrib.get('type'),
                   'sarcasm': link.attrib.get('sarcasm'),
                   'polarity': link.attrib.get('polarity')}

            src = link.find('source')
            if src is not None:
                res['src'] = src.attrib['ere_id']

            yield res


def extract_sent(annotations):
    for trg in annotations:

        links = trg.find('sentiments').getchildren()

        for link in links:

            res = {'trg': trg.attrib['ere_id'],
                   'sarcasm': link.attrib.get('sarcasm'),
                   'polarity': link.attrib.get('polarity')}

            src = link.find('source')
            if src is not None:
                res['src'] = src.attrib['ere_id']

            if trg.tag == 'entity':
                assert trg.find('text') is not None
                res['trg_mention'] = trg.find('text').text

            yield res


# All functions below were only used in the exploration scripts
# but can be used if you find them useful.

# arguments (to events/relations) can be either entity mentions or fillers.
#   (I haven't yet found any exceptions.)

# This function checks whether an argument is an entity or a filler,
# and finds its offset and length

def argument_offset(arg, entity_mentions, fillers):
    if 'entity_mention_id' in arg.attrib:
        arg_ent = entity_mentions[arg.attrib['entity_mention_id']]
        offset = int(arg_ent['mention_offset'])
        length = int(arg_ent['mention_length'])
    elif 'filler_id' in arg.attrib:
        filler = fillers[arg.attrib['filler_id']]
        offset = int(filler['offset'])
        length = int(filler['length'])
    else:
        print(arg.attrib)
        return None
    return offset, length

# Added by ANUSHA
def argument_token_number(arg,entity_mentions,fillers):
    if 'entity_mention_id' in arg.attrib:
        arg_ent = entity_mentions[arg.attrib['entity_mention_id']]
        token_num = int(arg_ent['mention_offset'])
        length = int(arg_ent['mention_length'])
    elif 'filler_id' in arg.attrib:
        filler = fillers[arg.attrib['filler_id']]
        token_num = int(filler['offset'])
        length = int(filler['length'])
    else:
        print(arg.attrib)
        return None
    return token_num,length

# This long function parses the ERE file into dicts, indexed by mention ids

def extract_mentions(ere_root):
    entity_mentions = {}
    fillers = {}
    relation_mentions = {}
    event_mentions = {}

    # entities

    for entity in ere_root.find('entities').getchildren():

        for mention in entity.getchildren():
            mention_props = {'mention_%s' % key: mention.attrib[key]
                             for key in ['noun_type', 'length', 'offset']}
            mention_props['text'] = mention.find('mention_text').text
            mention_props['id'] = entity.attrib['id']
            mention_props['entity_type'] = entity.attrib['type']

            entity_mentions[mention.attrib['id']] = mention_props

    # fillers
    fillers_obj = ere_root.find('fillers')
    if fillers_obj is not None:
        for filler in fillers_obj.getchildren():
            fillers[filler.attrib['id']] = {key: filler.attrib[key]
                                            for key
                                            in ['length', 'offset', 'type']}

    # relations

    for relation in ere_root.find('relations').getchildren():
        for mention in relation.getchildren():

            positions = []
            arg1 = mention.find('rel_arg1')
            arg2 = mention.find('rel_arg2')
            trigger = mention.find('trigger')

            if trigger is not None:
                positions.append(int(trigger.attrib['offset']))
                positions.append(int(trigger.attrib['offset']) +
                                 int(trigger.attrib['length']))

            for arg in (arg1, arg2):
                offset, length = argument_offset(arg, entity_mentions, fillers)
                positions.append(offset)
                positions.append(offset + length)

            offset = min(positions)
            length = max(positions) - offset + 1

            mention_props = {'id': relation.attrib['id'],
                             'mention_offset': offset,
                             'mention_length': length}

            relation_mentions[mention.attrib['id']] = mention_props

    # events

    for event in ere_root.find('hoppers').getchildren():
        hopper_id = event.attrib['id']
        for mention in event.getchildren():
            positions = []

            trigger = mention.find('trigger')
            if trigger is not None:
                positions.append(int(trigger.attrib['offset']))
                positions.append(int(trigger.attrib['offset']) +
                                 int(trigger.attrib['length']))

            for arg in mention.findall('em_arg'):
                offset, length = argument_offset(arg, entity_mentions, fillers)
                positions.append(offset)
                positions.append(offset + length)

            offset = min(positions)
            length = max(positions) - offset + 1

            mention_props = {'id': hopper_id,
                             'mention_offset': offset,
                             'mention_length': length}

            event_mentions[mention.attrib['id']] = mention_props

    return entity_mentions, relation_mentions, event_mentions


# This long function parses the ERE file into dicts, indexed by mention ids

def extract_mentions_pred(ere_root):
    entity_mentions = {}
    fillers = {}
    relation_mentions = {}
    event_mentions = {}

    # entities

    for entity in ere_root.find('entities').getchildren():

        for mention in entity.getchildren():
            mention_props = {'mention_%s' % key: mention.attrib[key]
                             for key in ['noun_type', 'length', 'offset']}
            mention_props['text'] = mention.find('mention_text').text
            mention_props['id'] = entity.attrib['id']
            mention_props['entity_type'] = entity.attrib['type']

            entity_mentions[mention.attrib['id']] = mention_props

    # fillers

    '''for filler in ere_root.find('fillers').getchildren():
        fillers[filler.attrib['id']] = {key: filler.attrib[key]
                                        for key
                                        in ['length', 'offset', 'type']}'''

    # relations

    for relation in ere_root.find('relations').getchildren():
        for mention in relation.getchildren():

            positions = []
            arg1 = mention.find('rel_arg1')
            arg2 = mention.find('rel_arg2')
            trigger = mention.find('trigger')

            if trigger is not None:
                positions.append(int(trigger.attrib['offset']))
                positions.append(int(trigger.attrib['offset']) +
                                 int(trigger.attrib['length']))

            for arg in (arg1, arg2):
                offset, length = argument_offset(arg, entity_mentions, fillers)
                positions.append(offset)
                positions.append(offset + length)

            offset = min(positions)
            length = max(positions) - offset + 1

            mention_props = {'id': relation.attrib['id'],
                             'mention_offset': offset,
                             'mention_length': length}

            relation_mentions[mention.attrib['id']] = mention_props

    # events

    for event in ere_root.find('hoppers').getchildren():
        hopper_id = event.attrib['id']
        for mention in event.getchildren():
            positions = []

            trigger = mention.find('trigger')
            if trigger is not None:
                positions.append(int(trigger.attrib['offset']))
                positions.append(int(trigger.attrib['offset']) +
                                 int(trigger.attrib['length']))

            for arg in mention.findall('em_arg'):
                offset, length = argument_offset(arg, entity_mentions, fillers)
                positions.append(offset)
                positions.append(offset + length)

            offset = min(positions)
            length = max(positions) - offset + 1

            mention_props = {'id': hopper_id,
                             'mention_offset': offset,
                             'mention_length': length}

            event_mentions[mention.attrib['id']] = mention_props

    return entity_mentions, relation_mentions, event_mentions
