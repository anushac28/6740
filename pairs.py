from itertools import product
from lxml import etree as et
from loaders import extract_mentions, extract_sent, extract_belief, extract_mentions_pred
from collections import Counter
from loaders import iter_best_split


def pairs_from_doc(best_file, strict_neg=False):

    _mentions = extract_mentions(best_file.ere)

    entity_mentions, relation_mentions, event_mentions = _mentions

    entity_mention_keys = sorted(list(entity_mentions.keys()))
    relation_mention_keys = sorted(list(relation_mentions.keys()))
    event_mention_keys = sorted(list(event_mentions.keys()))

    # beliefs = best_file.annotation.find('belief_annotations')
    sentiments = best_file.annotation.find('sentiment_annotations')

    # be_rel = beliefs.find('relations').getchildren()
    # be_evt = beliefs.find('events').getchildren()

    st_ent = sentiments.find('entities').getchildren()
    st_rel = sentiments.find('relations').getchildren()
    st_evt = sentiments.find('events').getchildren()

    belief_links = {}
    sent_links = {}

    belief_obj_links = set()
    sent_obj_links = set()

    # Let's consider "None" a mention of a source called "None" itself.
    obj_id = {k: v.attrib['id'] for k, v in best_file.objects.items()}
    obj_id[None] = "None"

    # for link in extract_belief(be_rel + be_evt):
    #     src = link.get('src')  # None if missing
    #     trg = link['trg']
    #     belief_links[src, trg] = {
    #         k: v for k, v in link.items()
    #         if k in ['belief_type', 'polarity', 'sarcasm']
    #     }

    #     if link['belief_type'] != 'na':
    #         belief_obj_links.add((obj_id[src], obj_id[trg]))

    for link in extract_sent(st_ent + st_rel + st_evt):
        src = link.get('src')  # None if missing
        trg = link['trg']
        sent_links[src, trg] = {
            k: v for k, v in link.items()
            if k in ['polarity', 'sarcasm']
        }

        if link['polarity'] != 'none':
            sent_obj_links.add((obj_id[src], obj_id[trg]))

    sent_candidates = product([None] + entity_mention_keys,
                              entity_mention_keys +
                              relation_mention_keys +
                              event_mention_keys)

    # belief_candidates = product([None] + entity_mention_keys,
    #                             relation_mention_keys + event_mention_keys)

    sent_pairs = []
    belief_pairs = []

    for src, trg in sent_candidates:
        # src == trg is actually possible in rare cases!
        # m-273 in 24d93564f48ae17904aa82f937db8c21.best.xml
        if src == trg:
            continue

        y = sent_links.get((src, trg))

        if strict_neg and (y is None or y['polarity'] == 'none'):
            # is this missing link between things that have some link?
            if (obj_id[src], obj_id[trg]) in sent_obj_links:
                continue

        sent_pairs.append((src, trg, y))

    # for src, trg in belief_candidates:

    #     y = belief_links.get((src, trg))

    #     if strict_neg and (y is None or y['belief_type'] == 'na'):
    #         # is this missing link between things that have some link?
    #         if (obj_id[src], obj_id[trg]) in sent_obj_links:
    #             continue

    #     belief_pairs.append((src, trg, y))

    return sent_pairs #, belief_pairs


def pairs_from_doc_eval(best_file, strict_neg=False):
    _mentions = extract_mentions(best_file.ere)

    entity_mentions, relation_mentions, event_mentions = _mentions

    entity_mention_keys = sorted(list(entity_mentions.keys()))
    relation_mention_keys = sorted(list(relation_mentions.keys()))
    event_mention_keys = sorted(list(event_mentions.keys()))

    belief_links = {}
    sent_links = {}

    belief_obj_links = set()
    sent_obj_links = set()

    # Let's consider "None" a mention of a source called "None" itself.
    obj_id = {k: v.attrib['id'] for k, v in best_file.objects.items()}
    obj_id[None] = "None"

    sent_pairs = []
    belief_pairs = []

    sent_candidates = product([None] + entity_mention_keys,
                              entity_mention_keys +
                              relation_mention_keys +
                              event_mention_keys)

    #belief_candidates = product([None] + entity_mention_keys, relation_mention_keys + event_mention_keys)


    for src, trg in sent_candidates:
        # src == trg is actually possible in rare cases!
        # m-273 in 24d93564f48ae17904aa82f937db8c21.best.xml
        if src == trg:
            continue

        '''y = sent_links.get((src, trg))

        if strict_neg and (y is None or y['polarity'] == 'none'):
            # is this missing link between things that have some link?
            if (obj_id[src], obj_id[trg]) in sent_obj_links:
                continue'''

        sent_pairs.append((src, trg, -1))

    # for src, trg in belief_candidates:

    #     '''y = belief_links.get((src, trg))

    #     if strict_neg and (y is None or y['belief_type'] == 'na'):
    #         # is this missing link between things that have some link?
    #         if (obj_id[src], obj_id[trg]) in sent_obj_links:
    #             continue'''

    #     belief_pairs.append((src, trg, -1))

    return sent_pairs  #, belief_pairs


def xml_from_pairs(sent_pairs, belief_pairs):
    doc = et.Element("committed_belief_doc")
    # beliefs = et.SubElement(doc, "belief_annotations")
    # bel_rel = et.SubElement(beliefs, "relations")
    # bel_evt = et.SubElement(beliefs, "events")

    sentiments = et.SubElement(doc, "sentiment_annotations")
    sent_ent = et.SubElement(sentiments, "entities")
    sent_rel = et.SubElement(sentiments, "relations")
    sent_evt = et.SubElement(sentiments, "events")

    name = dict(relm="relation", em="event", m="entity")
    # bel_parent = dict(relm=bel_rel, em=bel_evt)
    sent_parent = dict(relm=sent_rel, em=sent_evt, m=sent_ent)

    # for src, trg, attrib in belief_pairs:
    #     if attrib is not None:

    #         # FIXME: the below modifies the pair list inplace!!
    #         belief_type = attrib.pop("belief_type")
    #         attrib["type"] = belief_type
    #         attrib = {k: v for k, v in attrib.items() if v is not None}
    #         kind = trg.split("-", 1)[0]
    #         t = et.SubElement(bel_parent[kind], name[kind], ere_id=trg)
    #         ann = et.SubElement(et.SubElement(t, "beliefs"), "belief",
    #                             **attrib)
    #         if src is not None:
    #             et.SubElement(ann, "source", ere_id=src)

    for src, trg, attrib in sent_pairs:
        if attrib is not None and attrib["polarity"] is not None:
            attrib = {k: v for k, v in attrib.items() if v is not None}
            kind = trg.split("-", 1)[0]
            t = et.SubElement(sent_parent[kind], name[kind], ere_id=trg)
            ann = et.SubElement(et.SubElement(t, "sentiments"), "sentiment",
                                **attrib)

            if src is not None:
                et.SubElement(ann, "source", ere_id=src)

    return et.tostring(doc, pretty_print=True)


if __name__ == '__main__':

    DATA_ROOT = '/home/anusha/Desktop/TRIALS/data'
    for doc in iter_best_split(DATA_ROOT, 'train'):
        sent_pairs = pairs_from_doc(doc, strict_neg=True)
        print(len(sent_pairs))
        print(sent_pairs)
        print(Counter(y["polarity"] if y is not None else "none"
                      for _, _, y in sent_pairs))
        #print(len(belief_pairs))
        #print(Counter(y["belief_type"] if y is not None else "na" for _, _, y in belief_pairs))

        break
