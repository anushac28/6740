from itertools import product

from loaders import iter_best_split
from loaders import _BaseBestFile
from pairs import pairs_from_doc
from distances import span_distance, find_closest_mention_by_entity, find_closest_author_by_entity, find_closest_mention_by_entity_left

from loaders import extract_mentions, extract_sent, extract_belief

import operator

import cPickle
import gzip
import os
import operator

import numpy
import theano



def prepare_data(seqs, maxlen=None) :
    lengths = [len(s) for s in seqs]

    if maxlen is not None :
        new_seqs = []
        new_labels = []
        new_lengths = []

        for l, s in zip(lengths, seqs) : 
            if l < maxlen :
                new_seqs.append(s)
                new_lengths.append(l)

        lengths = new_lengths
        seqs = new_seqs

        if len(lengths) < 1 :
            return None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int32')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)

    for idx, s in enumerate(seqs) :
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1

    return x, x_mask


def prepare_sent_separator(seqs, maxlen=None) :
    lengths = [len(s) for s in seqs]

    if maxlen is not None :
        new_seqs = []
        new_labels = []
        new_lengths = []

        for l, s in zip(lengths, seqs) : 
            if l < maxlen :
                new_seqs.append(s)
                new_lengths.append(l)

        lengths = new_lengths
        seqs = new_seqs

        if len(lengths) < 1 :
            return None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples, 2)).astype('int32')
    #x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)

    for idx, s in enumerate(seqs) :
        x[:lengths[idx], idx] = s
        #x_mask[:lengths[idx], idx] = 1

    return x


def prepare_data_relations(seqs, labels, maxlen=None) :
    lengths = [len(s) for s in seqs]

    if maxlen is not None :
        new_seqs = []
        new_labels = []
        new_lengths = []

        for l, s, y in zip(lengths, seqs, labels) :
            if l < maxlen :
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
            lengths = new_lengths
            labels = new_labels
            seqs = new_seqs

            if len(lengths) < 1 :
                return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    changed = 0

    if maxlen == 0 :
        changed = 1
        maxlen = 1

    x = numpy.zeros((maxlen, n_samples, 4)).astype('int32')
    x_mask = numpy.zeros((maxlen, n_samples, 4)).astype(theano.config.floatX)
    labels_new = numpy.zeros((maxlen, n_samples)).astype('int32')

    if changed == 0 :
        for idx, s in enumerate(seqs) :
            #print str(s)+"\t"+str(idx)
            #exit()
            x[:lengths[idx], idx] = s
            x_mask[:lengths[idx], idx] = 1.

            for idx, y in enumerate(labels) :
                labels_new[:lengths[idx], idx] = y

    return x, x_mask, labels_new
    


def prepare_data_relations_all(seqs_start, seqs_start_ind, seqs_end, seqs_end_ind, labels, maxlen=None) :
    lengths = [len(s) for s in seqs_start]

    if maxlen is not None :
        new_seqs = []
        new_labels = []
        new_lengths = []

        for l, s, y in zip(lengths, seqs_start, labels) :
            if l < maxlen :
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
            lengths = new_lengths
            labels = new_labels
            seqs = new_seqs

            if len(lengths) < 1 :
                return None, None, None

    n_samples = len(seqs_start)
    maxlen = numpy.max(lengths)

    maxstart = 0
    maxend = 0
    for seq in seqs_start[0] : 
        if len(seq) > maxstart : 
            maxstart = len(seq)
            
    for seq in seqs_end[0] : 
        if len(seq) > maxend : 
            maxend = len(seq)
        
    changed = 0

    if maxlen == 0 :
        changed = 1
        maxlen = 1

    x_start = numpy.zeros((maxlen, n_samples, maxstart)).astype('int32')
    x_start_ind = numpy.zeros((maxlen, n_samples, 1)).astype('int32')

    x_end = numpy.zeros((maxlen, n_samples, maxend)).astype('int32')
    x_end_ind = numpy.zeros((maxlen, n_samples, 1)).astype('int32')

    x_start_mask = numpy.zeros((maxlen, n_samples, maxstart)).astype(theano.config.floatX)
    x_end_mask = numpy.zeros((maxlen, n_samples, maxend)).astype(theano.config.floatX)

    labels_new = numpy.zeros((maxlen, n_samples)).astype('int32')

    '''print seqs_start
    print maxstart
    print lengths'''

    if changed == 0 :

        for idx, s in enumerate(seqs_start_ind) :
            x_start_ind[:lengths[idx], idx] = s

        for idx, s in enumerate(seqs_end_ind) :
            x_end_ind[:lengths[idx], idx] = s


        for idx, s in enumerate(seqs_start) :
            #print str(s)+"\t"+str(idx)
            #print x_start[:lengths[idx], idx]
            for idy, entry in enumerate(s) : 
                #print idy
                for idz, entry1 in enumerate(entry) :
                    x_start[idy, idx, idz] = entry1
                #print x_start[idy, idx]
                x_start_mask[idy, idx] = 1.

        for idx, y in enumerate(labels) :
            labels_new[:lengths[idx], idx] = y

        for idx, s in enumerate(seqs_end) :
            #print str(s)+"\t"+str(idx)
            #print x_start[:lengths[idx], idx]
            for idy, entry in enumerate(s) : 
                #print idy
                for idz, entry1 in enumerate(entry) :
                    x_end[idy, idx, idz] = entry1
                #print x_end[idy, idx]
                x_end_mask[idy, idx] = 1.

    #print x_start

    return x_start, x_start_ind, x_start_mask, x_end, x_end_ind, x_end_mask, labels_new



def prepare_data_relations_target(relations_targets, relations_holders, relations_y1, lengths, maxlen=None) :

    maxlen = -1
    for i in numpy.arange(len(relations_holders)) : 
        relation_holder = relations_holders[i]
        if maxlen <  len(relation_holder) : 
            maxlen = len(relation_holder)

    relations_targets1 = numpy.zeros((len(relations_targets), 2)).astype('int32')
    relations_holders1 = numpy.zeros((len(relations_holders), maxlen, 2)).astype('int32')
    relations_y2 = numpy.zeros((len(relations_holders), maxlen, 1)).astype('int32')
    lengths1 = numpy.zeros((len(relations_holders), 1)).astype('int32')

    lengthss = [len(s) for s in relations_targets]

    print lengthss

    for idx, s in enumerate(relations_targets) :
        relations_targets1[idx, :lengthss[idx]] = s
        
    for idx, s in enumerate(lengths) :
        lengths1[idx,] = s

    for idx, s in enumerate(relations_holders) :
        for idy, entry in enumerate(s) : 
            for idz, entry1 in enumerate(entry) :
                relations_holders1[idx, idy, idz] = entry1

    for idx, s in enumerate(relations_y1) :
        for idy, entry in enumerate(s) : 
                relations_y2[idx, idy] = entry


    return relations_targets1, relations_holders1, relations_y2, lengths1



def _coord(mention):
    """Returns the integer coordinates of a mention: the offset and length"""
    return int(mention.attrib['offset']), int(mention.attrib['length'])


def _bound(coords):
    """Gets a tight (offset, length) to encompass all arguments

    eg: _bound((1, 3), (2, 4)) == (1, 5)

    """
    starts = [offset for offset, _ in coords]
    ends = [offset + length for offset, length in coords]

    offset = min(starts)
    end = max(ends)
    length = end - offset

    return offset, length


def doc_features(doc):
    return {'domain': doc.data_type}

def src_features(doc, src_id):

    if src_id is None:
        return {'src_missing': 'true'}
    else:
        out = {'src_{}'.format(key): val
               for key, val in entity_features(doc, src_id).items()}
        out['src_missing'] = 'false'

        return out

def trg_features(doc, trg_id):
    # delegate depending on target type
    select = dict(relm=relation_features, em=event_features, m=entity_features)
    prefix = trg_id.split('-')[0]
    func = select[prefix]
    out = func(doc, trg_id)
    return {'trg_{}'.format(key): val for key, val in out.items()}

def _mention_toks(doc, mention):
    offset, length = _coord(mention)
    start, end = doc.offset_to_tokens(offset, length)

    if start is None:
        # mention is inside some XML tag, e.g. forum username.
        # Generate dummy but compatible-ish conll
        res = [[-1, mention.find('mention_text').text, '_', "NOUN", "NNP", "_",
                "_", "_", "_", "_"]]

    else:
        res = doc.parsed_span(start, end)
    return start, end, res




def relation_features(doc, mention_id):
    relation = doc.objects[mention_id]
    mention = doc.mentions[mention_id]

    # always two argument entities

    arg1 = mention.find('rel_arg1')
    arg2 = mention.find('rel_arg2')

    if 'entity_mention_id' in arg1.attrib:
        arg1 = arg1.attrib['entity_mention_id']
    else:
        arg1 = arg1.attrib['filler_id']

    if 'entity_mention_id' in arg2.attrib:
        arg2 = arg2.attrib['entity_mention_id']
    else:
        arg2 = arg2.attrib['filler_id']

    arg1_mention = doc.mentions[arg1]
    arg2_mention = doc.mentions[arg2]
    arg1_offset, arg1_length = _coord(arg1_mention)
    arg2_offset, arg2_length = _coord(arg2_mention)

    offset, length = _bound([(arg1_offset, arg1_length),
                             (arg2_offset, arg2_length)])

    start, end = doc.offset_to_tokens(offset, length)

    res = {'type': 'relation',
           'relation_type': relation.attrib['type'],
           'relation_subtype': relation.attrib['subtype'],
           'mention_realization': mention.attrib['realis'],
           'offset_length_': (offset, length)

           }

    if start is not None:
        span = doc.parsed_span(start, end)

        res['mention_text_bow'] = [[w[1] for w in span]]
        res['mention_pos_bow'] = [[w[3] for w in span]]
        res['sentences_'] = (start[0], end[0])
        res['tok_ids_'] = (start[1], end[1])

    return res

def event_features(doc, mention_id):
    event = doc.objects[mention_id]
    mention = doc.mentions[mention_id]

    # todo: include props and text of event arguments
    args = mention.findall('em_arg')  # todo: each arg has a role, use it?

    coords = []
    for arg in args:
        if 'entity_mention_id' in arg.attrib:
            m = doc.mentions[arg.attrib['entity_mention_id']]
        else:
            m = doc.mentions[arg.attrib['filler_id']]
        coords.append(_coord(m))

    trigger = mention.find('trigger')
    if trigger is not None:
        coords.append(_coord(trigger))

    offset, length = _bound(coords)

    start, end = doc.offset_to_tokens(offset, length)

    res = {'type': 'event',
           'mention_event_type': mention.attrib['type'],
           'mention_event_subtype': mention.attrib['subtype'],
           'mention_realization': mention.attrib['realis'],
           'offset_length_': (offset, length)
           }

    if start is not None:
        span = doc.parsed_span(start, end)

        res['mention_text_bow'] = [[w[1] for w in span]]
        res['mention_pos_bow'] = [[w[3] for w in span]]

        res['sentences_'] = (start[0], end[0])
        res['tok_ids_'] = (start[1], end[1])

    return res


def entity_features(doc, mention_id):
    entity = doc.objects[mention_id]
    mention = doc.mentions[mention_id]

    start, end, span = _mention_toks(doc, mention)
    mention_text = [w[1] for w in span]

    mention_pos = [w[3] for w in span]

    all_mentions_text = []
    all_mentions_pos = []
    for m in entity.findall('entity_mention'):
        _, _, span = _mention_toks(doc, m)
        all_mentions_text.append([w[1] for w in span])
        all_mentions_pos.append([w[3] for w in span])

    res = {'type': 'entity',
           'mention_text_bow': [mention_text],
           'mention_pos_bow': [mention_pos],
           'mention_entity_noun_type': mention.attrib['noun_type'],
           'entity_type': entity.attrib['type'],
           'entity_specificity': entity.attrib['specificity'],
           'all_mentions_text_bow': all_mentions_text,
           'all_mentions_pos_bow': all_mentions_pos,
           'offset_length_': _coord(mention)
           }

    if start is not None:
        res['sentences_'] = (start[0], end[0])
        res['tok_ids_'] = (start[1], end[1])

    return res


def read_tokenized(tokenized) : 

    start_docs = dict()
    start_tokens = dict()
    end_docs = dict()
    end_tokens = dict()

    sent_split_dict = dict()
    sent_split = []

    current_doc = None
    current_sent = -1
    current_token = -1
    start_token = -1
    end_token = -1

    tokens_f = open("tokenized.txt")
    for line in tokens_f : 
        if len(line.split("\t")) == 1 : 
            if ".txt" in line.strip() or ".xml" in line.strip(): 
                if current_doc is not None : 
                    start_docs[current_doc] = start_tokens
                    end_docs[current_doc] = end_tokens
                    sent_split_dict[current_doc] = sent_split
                if ".txt" in line.strip() : 
                    current_doc = line.strip()[51:-8]
                if ".xml" in line.strip() : 
                    current_doc = line.strip()[51:-4]
                start_tokens = dict()
                end_tokens = dict()
                sent_split = []
                current_token = -1
                current_sent = -1
                start_token = -1
                end_token = -1
        else : 
            current_token += 1
            start_tokens[int(line.split("\t")[0])] = line.split("\t")[2].strip()
            end_tokens[int(line.split("\t")[0])] = int(line.split("\t")[1].strip())
            if current_sent == int(line.split("\t")[3].strip()) : 
                end_token = current_token
            else : 
                if start_token != -1 and end_token != -1 : 
                    sent_split.append([start_token, end_token])
                current_sent = int(line.split("\t")[3].strip())
                start_token = current_token
                end_token = current_token

    if len(start_tokens.keys()) > 0 : 
        start_docs[current_doc] = start_tokens
        end_docs[current_doc] = end_tokens
        sent_split_dict[current_doc] = sent_split
        
    #print len(docs.keys())
    #print sent_split_dict

    return start_docs, end_docs, sent_split_dict


def get_index(start_docs, end_docs, doc, offset, length) : 
    start_doc_tokens = start_docs[doc]
    end_doc_tokens = end_docs[doc]

    #print start_doc_tokens

    sorted_indices = sorted([int(key) for key in start_doc_tokens.keys()])

    
    index_dict = dict()
    count = 1
    for index in sorted_indices : 
        index_dict[index] = count
        count+=1

    curr_len = offset

    nearest_offset = 0
    if offset not in index_dict.keys() : 
        for ind in sorted_indices : 
            #print str(ind)+"\t"+str(start_doc_tokens[ind])
            #find the nearest key to the left and return
            if ind < offset and (offset-ind) < (offset-nearest_offset) : 
                nearest_offset = ind
        offset = nearest_offset
        #print "Changed\t"+str(index_dict[offset])
        #print offset

    begin_index = index_dict[offset]

    end_index = index_dict[offset]

    curr_len = end_doc_tokens[offset]

    '''print "======"
    print offset
    print length'''

    while curr_len < offset+length-1 : 

        if curr_len not in start_doc_tokens.keys() and curr_len+1 in start_doc_tokens.keys(): 
            curr_len += 1
        '''if curr_len not in start_doc_tokens.keys() and curr_len+2 in start_doc_tokens.keys(): 
            curr_len += 2
        if curr_len not in start_doc_tokens.keys() and curr_len+3 in start_doc_tokens.keys(): 
            curr_len += 3
        if curr_len not in start_doc_tokens.keys() and curr_len+4 in start_doc_tokens.keys(): 
            curr_len += 4'''

        ''' if curr_len not in start_doc_tokens.keys() and curr_len-1 in start_doc_tokens.keys(): 
            curr_len -= 1'''
        
        '''if curr_len not in start_doc_tokens.keys() : 
            print "--"+str(curr_len)'''
        if curr_len not in start_doc_tokens.keys() : 
            print begin_index
            print end_index
            for ind in sorted_indices : 
                print str(ind)+"\t"+str(start_doc_tokens[ind])
            print "==="
            print str(offset)+"\t"+str(length)
        word = start_doc_tokens[curr_len]
        
        #if doc == '01f69c4c2206e7c3fa3706ccd5b8b350' : 
        #print str(curr_len)+"\t"+str(offset+length)+"\t"+str(word)+"\t"+str(end_doc_tokens[curr_len])


        #print word
        end_index = index_dict[curr_len]
        #curr_len += len(word)

        if curr_len == end_doc_tokens[curr_len] : 
            curr_len += 1
        else : 
            curr_len = end_doc_tokens[curr_len]
        

        #print "++"+str(curr_len)

    return (begin_index, end_index)


def Newpairs_from_doc(best_file) : 

    '''print best_file.doc_id
    print best_file.source[3583:3583+3]
    print best_file.source'''
    
    #print best_file.ere
    #exit()
    
    _mentions = extract_mentions(best_file.ere)
    entity_mentions, relation_mentions, event_mentions = _mentions

    entity_mention_keys = sorted(list(entity_mentions.keys()))
    relation_mention_keys = sorted(list(relation_mentions.keys()))
    event_mention_keys = sorted(list(event_mentions.keys()))

    sentiments = best_file.annotation.find('sentiment_annotations')

    st_ent = sentiments.find('entities').getchildren()
    st_rel = sentiments.find('relations').getchildren()
    st_evt = sentiments.find('events').getchildren()
    
    '''print len(st_ent)
    print len(st_rel)
    print len(st_evt)'''

    sent_links = {}

    count11 = 0
    for link in extract_sent(st_ent + st_rel + st_evt) : 
        count11 += 1
        src = link.get('src') #None if missing
        trg = link['trg']
        #print src

        features = {}
        src_feats = src_features(best_file, src)
        features.update(src_feats)

        trg_feats = trg_features(best_file, trg)
        features.update(trg_feats)

        #src_offset, src_length = features.get('src_offset_length_', (0, 0))
        trg_offset, trg_length = features['trg_offset_length_']

        if src is not None : 
            '''ent_id = doc.objects[src_id]
            key = 'id'
            print ent_id.values()[0]'''
            
            src_offset_nearest = None
            src_length_nearest = None
            src_id_nearest = None

            src_offset_author = None
            src_length_author = None
            src_id_author = None

            src_offset_left = None
            src_length_left = None
            src_id_left = None


            min_dist, nearest_nearest = find_closest_mention_by_entity(best_file.objects[src],
                                                        trg_offset, trg_length)
            exit_case = 0

            if nearest_nearest is not None : 
                src_offset_nearest = nearest_nearest.attrib['offset']
                src_length_nearest = nearest_nearest.attrib['length']
                src_id_nearest = nearest_nearest.attrib['id']

                sent_links[src_id_nearest, trg] = {k : v for k, v in link.items()
                                                   if k in ['polarity', 'sarcasm']
                                               }

                '''if str(src_offset_nearest) > str(trg_offset) : 
                    if int(src_offset_nearest) - int(trg_offset) > 25 : 
                        #print str(src_offset_nearest)+"\t"+str(trg_offset)
                        exit_case =1'''
            
            min_dist, nearest_author = find_closest_author_by_entity(best_file, best_file.objects[src],
                                                              trg_offset, trg_length)

            if nearest_author is not None : 
                src_offset_author = nearest_author.attrib['offset']
                src_length_author = nearest_author.attrib['length']
                src_id_author = nearest_author.attrib['id']
           
                if not (src_id_nearest == src_id_author) : 
                    sent_links[src_id_author, trg] = {k : v for k, v in link.items()
                                                      if k in ['polarity', 'sarcasm']
                                                  }
                
                    '''if exit_case == 1 : 
                        print str(src_offset_author)+"\t"+str(trg_offset)
                        print "-----"'''
            
            min_dist, nearest_left = find_closest_mention_by_entity_left(best_file.objects[src],
                                                        trg_offset, trg_length)

            if nearest_left is not None : 
                src_offset_left = nearest_left.attrib['offset']
                src_length_left = nearest_left.attrib['length']
                src_id_left = nearest_left.attrib['id']

  
                if not (src_id_nearest == src_id_left) and not (src_id_left == src_id_author) :
                    sent_links[src_id_left, trg] = {k : v for k, v in link.items()
                                                    if k in ['polarity', 'sarcasm']
                                                }

                    '''if exit_case == 1 : 
                        print str(src_offset_left)+"\t"+str(trg_offset)
                        print "====="'''


        '''else : 
            sent_links[src, trg] = {k : v for k, v in link.items()
                                               if k in ['polarity', 'sarcasm']
                                           }'''

    #print count11
    #exit()

    #Now create all the candidates : for all the entity keys (NOT entity mentions!!)

    sent_candidates = product([None], entity_mention_keys + relation_mention_keys + event_mention_keys)

    new_candidates = []

    for src, trg in sent_candidates : 
        new_candidates.append((src, trg))

    
    for src_entity in best_file.ere.find('entities') : 
        for trg in entity_mention_keys+relation_mention_keys+event_mention_keys : 

            features = {}
        
            trg_feats = trg_features(best_file, trg)
            features.update(trg_feats)

            trg_offset, trg_length = features['trg_offset_length_']
            

            src_offset_nearest = None
            src_length_nearest = None
            src_id_nearest = None

            src_offset_author = None
            src_length_author = None
            src_id_author = None
            
            src_offset_left = None
            src_length_left = None
            src_id_left = None


            '''print src_entity.attrib['id']
            print trg
            print best_file.source[:trg_offset]'''

            author_added = 0
            left_added = 0

            min_dist, nearest_author = find_closest_author_by_entity(best_file, src_entity,
                                                                     trg_offset, trg_length)

            if nearest_author is not None : 
                src_offset_author = nearest_author.attrib['offset']
                src_length_author = nearest_author.attrib['length']
                src_id_author = nearest_author.attrib['id']
            
                author_added = 1
                new_candidates.append((src_id_author, trg))



            min_dist, nearest_left = find_closest_mention_by_entity_left(src_entity,
                                                                         trg_offset, trg_length)

            if nearest_left is not None : 
                src_offset_left = nearest_left.attrib['offset']
                src_length_left = nearest_left.attrib['length']
                src_id_left = nearest_left.attrib['id']

                left_added = 1

                if not (src_id_left == src_id_author) :
                    new_candidates.append((src_id_left, trg))


            min_dist, nearest_nearest = find_closest_mention_by_entity(src_entity,
                                                        trg_offset, trg_length)
            if nearest_nearest is not None : 
                src_offset_nearest = nearest_nearest.attrib['offset']
                src_length_nearest = nearest_nearest.attrib['length']
                src_id_nearest = nearest_nearest.attrib['id']
                
                if not (src_id_nearest == src_id_author) and not (src_id_nearest == src_id_left) : 
                    (hobj1, hobj2) = doc.offset_to_tokens(src_offset_nearest, src_length_nearest)
                    (tobj1, tobj2) = doc.offset_to_tokens(trg_offset, trg_length)

                    if hobj1 is not None : 
                        (starth1, endh1) = hobj1
                    else : 
                        continue
                        #(starth1, endh1) = (0, 0)
                    
                    if hobj2 is not None : 
                        (starth2, endh2) = hobj2
                    else :
                        continue
                        #(starth2, endh2) = (0, 0)

                    
                    if tobj1 is not None : 
                        (startt1, endt1) = tobj1
                    else :
                        continue
                        #(startt1, endt1) = (0, 0)

                    if tobj2 is not None : 
                        (startt2, endt2) = tobj2
                    else :
                        continue
                        #(startt2, endt2) = (0, 0)

                    #print str(hobj1)+"\t"+str(hobj2)+"\t"+str(tobj1)+"\t"+str(tobj2)

                    if (left_added + author_added) > 0 or ((abs(starth2-startt2)<2)) :
                        if (src_offset_nearest > trg_offset) : 
                            print str(src_offset_nearest)+"\t"+str(trg_offset)+"\t"+str(starth2)+"\t"+str(startt2)+"\t"+str(hobj2)+"\t"+str(tobj2)
                        new_candidates.append((src_id_nearest, trg))
            
    #print sent_links
    sent_pairs = [(src, trg, sent_links.get((src, trg))) 
                  for (src, trg) in new_candidates
                  if src!=trg]

    return sent_pairs


def load_data(n_words=40000, maxlen=None) :
    
    import pickle
    import os
    
    DATA_ROOT = '/home/arzoo/Desktop/BeST_old/LDC2016E27_V2/data/'
    with open(os.path.join(DATA_ROOT, "sent_Traindata12.pkl"), "r") as f:
        (train_x, train_x_ind, train_y_relations, train_y_labels) = pickle.load(f)
    
    #with open(os.path.join(DATA_ROOT, "sent_Testdata112.pkl"), "r") as f:
    #(test_x, test_x_ind, test_y_relations, test_y_labels, test_y_ids) = pickle.load(f)
    with open(os.path.join(DATA_ROOT, "sent_Testdata12.pkl"), "r") as f:
        (test_x, test_x_ind, test_y_relations, test_y_labels) = pickle.load(f)

    
    start_tokenized_docs, end_tokenized_docs, sent_split_docs = read_tokenized('/home/arzoo/Desktop/BeST/BeSt/English/tokenized.txt')

    new_train_x=[]
    new_train_x_ind = []
    new_train_y_relations = []
    new_train_y_labels = []
    new_train_sent_split = []
    for doc_id in train_x.keys() : 
        new_train_x.append(train_x[doc_id])
        new_train_x_ind.append(train_x_ind[doc_id])
        new_train_y_relations.append(train_y_relations[doc_id])
        new_train_y_labels.append(train_y_labels[doc_id])
        new_train_sent_split.append(sent_split_docs[doc_id])

    new_test_x=[]
    new_test_x_ind = []
    new_test_y_relations = []
    new_test_y_labels = []
    new_test_sent_split = []
    #new_test_y_ids = []
    #doc_ids = []
    for doc_id in test_x.keys() :
        new_test_x.append(test_x[doc_id])
        new_test_x_ind.append(test_x_ind[doc_id])
        new_test_y_relations.append(test_y_relations[doc_id])
        new_test_y_labels.append(test_y_labels[doc_id])
        new_test_sent_split.append(sent_split_docs[doc_id])
        #new_test_y_ids.append(test_y_ids[doc_id])
        #doc_ids.append(doc_id)


    '''train = (new_train_x, new_train_x_ind, new_train_y_relations, new_train_y_labels)
    #test = (new_test_x, new_test_x_ind, new_test_y_relations, new_test_y_labels, new_test_y_ids, doc_ids)
    test = (new_test_x, new_test_x_ind, new_test_y_relations, new_test_y_labels)'''


    train = (new_train_x, new_train_x_ind, new_train_y_relations, new_train_y_labels, new_train_sent_split)
    #test = (new_test_x, new_test_x_ind, new_test_y_relations, new_test_y_labels, new_test_y_ids, doc_ids)
    test = (new_test_x, new_test_x_ind, new_test_y_relations, new_test_y_labels, new_test_sent_split)

    return train, test
    


    


def load_data_PosvsNeg(n_words=40000, maxlen=None) :
    
    import pickle
    import os
    
    DATA_ROOT = '/home/arzoo/Desktop/BeST/LDC2016E27_V2/data/'
    with open(os.path.join(DATA_ROOT, "sent_Traindata12.pkl"), "r") as f:
        (train_x, train_x_ind, train_y_relations, train_y_labels) = pickle.load(f)
    
    #with open(os.path.join(DATA_ROOT, "sent_Testdata112.pkl"), "r") as f:
    #(test_x, test_x_ind, test_y_relations, test_y_labels, test_y_ids) = pickle.load(f)
    with open(os.path.join(DATA_ROOT, "sent_Testdata12.pkl"), "r") as f:
        (test_x, test_x_ind, test_y_relations, test_y_labels) = pickle.load(f)

    
    start_tokenized_docs, end_tokenized_docs, sent_split_docs = read_tokenized('/home/arzoo/Desktop/BeST/BeSt/English/tokenized.txt')

    new_train_x=[]
    new_train_x_ind = []
    new_train_y_relations = []
    new_train_y_labels = []
    new_train_sent_split = []
    for doc_id in train_x.keys() : 
        #new_train_y_relations.append(train_y_relations[doc_id])
        #new_train_y_labels.append(train_y_labels[doc_id])
        #print train_y_labels[doc_id]
        #print train_y_relations[doc_id]
        new_set_labels = []
        new_set_relations = []
        for index in numpy.arange(len(train_y_labels[doc_id])) : 
            if train_y_labels[doc_id][index] == 1 or train_y_labels[doc_id][index] == 2 : 
                new_set_relations.append(train_y_relations[doc_id][index])
                new_set_labels.append(train_y_labels[doc_id][index]-1)

        if len(new_set_labels) != 0 :         
            new_train_x.append(train_x[doc_id])
            new_train_x_ind.append(train_x_ind[doc_id])
            
            new_train_y_relations.append(new_set_relations)
            new_train_y_labels.append(new_set_labels)
            
            new_train_sent_split.append(sent_split_docs[doc_id])

    new_test_x=[]
    new_test_x_ind = []
    new_test_y_relations = []
    new_test_y_labels = []
    new_test_sent_split = []
    #new_test_y_ids = []
    #doc_ids = []
    for doc_id in test_x.keys() :
        #new_test_y_relations.append(test_y_relations[doc_id])
        #new_test_y_labels.append(test_y_labels[doc_id])
        new_set_labels = []
        new_set_relations = []
        for index in numpy.arange(len(test_y_labels[doc_id])) : 
            if test_y_labels[doc_id][index] == 1 or test_y_labels[doc_id][index] == 2 : 
                new_set_relations.append(test_y_relations[doc_id][index])
                new_set_labels.append(test_y_labels[doc_id][index]-1)

        if len(new_set_labels) != 0 :         
            new_test_x.append(test_x[doc_id])
            new_test_x_ind.append(test_x_ind[doc_id])

            new_test_y_relations.append(new_set_relations)
            new_test_y_labels.append(new_set_labels)

            new_test_sent_split.append(sent_split_docs[doc_id])
        

    '''train = (new_train_x, new_train_x_ind, new_train_y_relations, new_train_y_labels)
    #test = (new_test_x, new_test_x_ind, new_test_y_relations, new_test_y_labels, new_test_y_ids, doc_ids)
    test = (new_test_x, new_test_x_ind, new_test_y_relations, new_test_y_labels)'''

    train = (new_train_x, new_train_x_ind, new_train_y_relations, new_train_y_labels, new_train_sent_split)
    #test = (new_test_x, new_test_x_ind, new_test_y_relations, new_test_y_labels, new_test_y_ids, doc_ids)
    test = (new_test_x, new_test_x_ind, new_test_y_relations, new_test_y_labels, new_test_sent_split)

    return train, test

#load_data_PosvsNeg()
#exit()



if __name__ == '__main__':

    import pickle
    import os

    start_tokenized_docs, end_tokenized_docs, sent_split_docs = read_tokenized('/home/arzoo/Desktop/BeST/BeSt/English/tokenized.txt')

    DATA_ROOT = '/home/arzoo/Desktop/BeST/LDC2016E27_V2/data/'

    X_sent = []
    y_sent = []
    doc_id_sent = []

    X_belief = []
    y_belief = []
    doc_id_belief = []

    print("extracting...")

    '''words_dict = dict()
    
    doc_count = 0

    #print start_tokenized_docs.keys()

    for doc in iter_best_split(DATA_ROOT, 'train'):

        #Find out why!!!

        doc_id = doc.doc_id

        if 'ENG_DF_000170_20150322_F00000082' in doc.doc_id : 
            doc_id = 'ENG_DF_000170_20150322_F00000082'
        if 'ENG_DF_000170_20150327_F0000007J' in doc.doc_id : 
            doc_id = 'ENG_DF_000170_20150327_F0000007J'
        if 'ENG_DF_000183_20150318_F0000009G' in doc.doc_id : 
            doc_id = 'ENG_DF_000183_20150318_F0000009G'
        if 'ENG_DF_000183_20150407_F0000009E' in doc.doc_id : 
            doc_id = 'ENG_DF_000183_20150407_F0000009E'
        if 'ENG_DF_000183_20150408_F0000009B' in doc.doc_id : 
            doc_id = 'ENG_DF_000183_20150408_F0000009B'
        if 'ENG_DF_000183_20150408_F0000009C' in doc.doc_id : 
            doc_id = 'ENG_DF_000183_20150408_F0000009C'
        if 'ENG_DF_000183_20150409_F0000009F' in doc.doc_id : 
            doc_id = 'ENG_DF_000183_20150409_F0000009F'
        if 'ENG_DF_000183_20150410_F0000009H' in doc.doc_id : 
            doc_id = 'ENG_DF_000183_20150410_F0000009H'
        if 'ENG_DF_000261_20150319_F00000084' in doc.doc_id : 
            doc_id = 'ENG_DF_000261_20150319_F00000084'
        if 'ENG_DF_000261_20150321_F00000081' in doc.doc_id : 
            doc_id = 'ENG_DF_000261_20150321_F00000081'
        

        doc_tokens = start_tokenized_docs[doc_id]

        print doc_id
        for tok in doc_tokens.keys() : 
            word = doc_tokens[tok]
            if word in words_dict :
                words_dict[word] = words_dict.get(word) + 1
            else :
                words_dict[word] = 1
            
    
    
    for doc in iter_best_split(DATA_ROOT, 'test'):
        doc_id = doc.doc_id

        if 'ENG_DF_000170_20150322_F00000082' in doc.doc_id :
            doc_id = 'ENG_DF_000170_20150322_F00000082'
        if 'ENG_DF_000170_20150327_F0000007J' in doc.doc_id :
            doc_id = 'ENG_DF_000170_20150327_F0000007J'
        if 'ENG_DF_000183_20150318_F0000009G' in doc.doc_id :
            doc_id = 'ENG_DF_000183_20150318_F0000009G'
        if 'ENG_DF_000183_20150407_F0000009E' in doc.doc_id :
            doc_id = 'ENG_DF_000183_20150407_F0000009E'
        if 'ENG_DF_000183_20150408_F0000009B' in doc.doc_id :
            doc_id = 'ENG_DF_000183_20150408_F0000009B'
        if 'ENG_DF_000183_20150408_F0000009C' in doc.doc_id :
            doc_id = 'ENG_DF_000183_20150408_F0000009C'
        if 'ENG_DF_000183_20150409_F0000009F' in doc.doc_id :
            doc_id = 'ENG_DF_000183_20150409_F0000009F'
        if 'ENG_DF_000183_20150410_F0000009H' in doc.doc_id :
            doc_id = 'ENG_DF_000183_20150410_F0000009H'
        if 'ENG_DF_000261_20150319_F00000084' in doc.doc_id :
            doc_id = 'ENG_DF_000261_20150319_F00000084'
        if 'ENG_DF_000261_20150321_F00000081' in doc.doc_id :
            doc_id = 'ENG_DF_000261_20150321_F00000081'

        doc_tokens = start_tokenized_docs[doc_id]
        print doc_id
        for tok in doc_tokens.keys() :
            word = doc_tokens[tok]
            if word in words_dict :
                words_dict[word] = words_dict.get(word) + 1
            else :
                words_dict[word] = 1

        
    print len(words_dict.keys())
    sorted_words_dict = sorted(words_dict.items(), key=operator.itemgetter(1))'''

    words_dict = dict()
    index_words_dict = dict()
    words_index_dict = dict()


    dict_f = open('dict.txt')

    for line in dict_f : 
     count = int(line.split("\t")[0])
     word = line.split("\t")[1].strip()
     index_words_dict[count] = word
     words_index_dict[word] = count

    unknown_count = count+1

    '''count = 0

    f_dict = open("dict.txt", 'w+')

    for key, value in sorted_words_dict :
        count +=1
        index_words_dict[count] = key
        words_index_dict[key] = count
        f_dict.write(str(count)+"\t"+str(key)+"\n")
   
    index_words_dict[count+1] = "NONE"
    words_index_dict["NONE"] = count+1
    f_dict.write(str(count+1)+"\t"+str("NONE")+"\n")

    
    f_dict.close()
    exit()'''

    train_x = dict()
    train_x_ind = dict()
    train_x_map = dict()

    doc_count = 0

    for doc in iter_best_split(DATA_ROOT, 'test'):
        print doc.doc_id

        doc_id = doc.doc_id

        if 'ENG_DF_000170_20150322_F00000082' in doc.doc_id : 
            doc_id = 'ENG_DF_000170_20150322_F00000082'
        if 'ENG_DF_000170_20150327_F0000007J' in doc.doc_id : 
            doc_id = 'ENG_DF_000170_20150327_F0000007J'
        if 'ENG_DF_000183_20150318_F0000009G' in doc.doc_id : 
            doc_id = 'ENG_DF_000183_20150318_F0000009G'
        if 'ENG_DF_000183_20150407_F0000009E' in doc.doc_id : 
            doc_id = 'ENG_DF_000183_20150407_F0000009E'
        if 'ENG_DF_000183_20150408_F0000009B' in doc.doc_id : 
            doc_id = 'ENG_DF_000183_20150408_F0000009B'
        if 'ENG_DF_000183_20150408_F0000009C' in doc.doc_id : 
            doc_id = 'ENG_DF_000183_20150408_F0000009C'
        if 'ENG_DF_000183_20150409_F0000009F' in doc.doc_id : 
            doc_id = 'ENG_DF_000183_20150409_F0000009F'
        if 'ENG_DF_000183_20150410_F0000009H' in doc.doc_id : 
            doc_id = 'ENG_DF_000183_20150410_F0000009H'
        if 'ENG_DF_000261_20150319_F00000084' in doc.doc_id : 
            doc_id = 'ENG_DF_000261_20150319_F00000084'
        if 'ENG_DF_000261_20150321_F00000081' in doc.doc_id : 
            doc_id = 'ENG_DF_000261_20150321_F00000081'
        
        doc_tokens = start_tokenized_docs[doc_id]
        #print doc_tokens
        
        train_x_doc = []
        train_x_doc_ind = []
        
        train_x_doc.append(words_index_dict["NNONE"])
        train_x_doc_ind.append(0)

        count = 1

        for key in sorted(doc_tokens.keys()) : 
            if doc_tokens[key] not in words_index_dict.keys() : 
                train_x_doc.append(unknown_count)
            else : 
                train_x_doc.append(words_index_dict[doc_tokens[key]])
            train_x_doc_ind.append(count)
            count+=1
       

        if doc_id not in train_x.keys() : 
            train_x[doc_id] = train_x_doc
            train_x_ind[doc_id] = train_x_doc_ind
        else : 
            initial = train_x[doc_id]
            initial += train_x_doc
            train_x[doc_id] = initial

            initial_ind = train_x_ind[doc_id]
            initial_ind += train_x_doc_ind
            train_x_ind[doc_id] = initial_ind


        #print train_x_doc_ind
        #print "==="

    #print train_x
    exit()
 
    train_y_relations = dict()
    train_y_labels = dict()
    train_y_ids = dict()


    all_authors = dict()
    doc_count = 0
    
    for doc in iter_best_split(DATA_ROOT, 'test') :
        #sent_pairs, belief_pairs = pairs_from_doc(doc)
        sent_pairs = Newpairs_from_doc(doc)

        #print sent_pairs
        #exit()

        doc_count += 1

        #doc_feats = doc_features(doc)
        cache = {}
        print doc.doc_id
        

        already_computed = dict()

        train_y_relations_doc = []
        train_y_labels_doc = []
        train_y_ids_doc = []
        
        author = dict()
        author_count = 1

        #print sent_pairs


        doc_id = doc.doc_id

        if 'ENG_DF_000170_20150322_F00000082' in doc.doc_id : 
            doc_id = 'ENG_DF_000170_20150322_F00000082'
        if 'ENG_DF_000170_20150327_F0000007J' in doc.doc_id : 
            doc_id = 'ENG_DF_000170_20150327_F0000007J'
        if 'ENG_DF_000183_20150318_F0000009G' in doc.doc_id : 
            doc_id = 'ENG_DF_000183_20150318_F0000009G'
        if 'ENG_DF_000183_20150407_F0000009E' in doc.doc_id : 
            doc_id = 'ENG_DF_000183_20150407_F0000009E'
        if 'ENG_DF_000183_20150408_F0000009B' in doc.doc_id : 
            doc_id = 'ENG_DF_000183_20150408_F0000009B'
        if 'ENG_DF_000183_20150408_F0000009C' in doc.doc_id : 
            doc_id = 'ENG_DF_000183_20150408_F0000009C'
        if 'ENG_DF_000183_20150409_F0000009F' in doc.doc_id : 
            doc_id = 'ENG_DF_000183_20150409_F0000009F'
        if 'ENG_DF_000183_20150410_F0000009H' in doc.doc_id : 
            doc_id = 'ENG_DF_000183_20150410_F0000009H'
        if 'ENG_DF_000261_20150319_F00000084' in doc.doc_id : 
            doc_id = 'ENG_DF_000261_20150319_F00000084'
        if 'ENG_DF_000261_20150321_F00000081' in doc.doc_id : 
            doc_id = 'ENG_DF_000261_20150321_F00000081'


        for src_id, trg_id, label in sent_pairs:
            #print str(author_count)+"\t"+str(len(sent_pairs))
            author_count += 1
            features = {}
            src_feats = src_features(doc, src_id)
            trg_feats = trg_features(doc, trg_id)
            
            features.update(src_feats)
            features.update(trg_feats)
            
            src_offset, src_length = features.get('src_offset_length_', (0, 0))
            trg_offset, trg_length = features['trg_offset_length_']

            if src_offset != 0  and src_length != 0 : 
                #if (src_offset, src_length) not in already_computed.keys() : 
                (hobj1, hobj2) = get_index(start_tokenized_docs, end_tokenized_docs, doc_id, src_offset, src_length)
                #already_computed[(src_offset, src_length)] = (hobj1, hobj2)
                #else : 
                #(hobj1, hobj2) = already_computed[(src_offset, src_length)]
            else : 
                (hobj1, hobj2) = (0, 0)
            
            
            if trg_offset != 0  and trg_length != 0 : 
                #if (trg_offset, trg_length) not in already_computed.keys() : 
                (tobj1, tobj2) = get_index(start_tokenized_docs, end_tokenized_docs, doc_id, trg_offset, trg_length)
                #    already_computed[(trg_offset, trg_length)] = (tobj1, tobj2)
                #else : 
                #    (tobj1, tobj2) = already_computed[(trg_offset, trg_length)]
            else : 
                (tobj1, tobj2) = (0, 0)
            
            #train_y_relations_doc.append([[hstart1, hend1, hstart2, hend2], [tstart1, tend1, tstart2, tend2]])


            if ([hobj1, hobj2, tobj1, tobj2]) not in train_y_relations_doc : 
                '''if tobj1 < hobj1 : 
                    print [hobj1, hobj2, tobj1, tobj2]
                    print str(src_offset)+"\t"+str(trg_offset)
                    print "----"'''
                train_y_relations_doc.append([hobj1, hobj2, tobj1, tobj2])

                #print "Added"

                if label is not None :
                    #train_y_labels_doc.append(label['polarity'])
                    if label['polarity'] == 'pos' :
                        train_y_labels_doc.append(1)
                        train_y_ids_doc.append((src_id, trg_id, label))
                        '''print str(src_id)+"\t"+str(trg_id)+"\t"+str(hobj1)+"\t"+str(hobj2)+"\t"+str(tobj1)+"\t"+str(tobj2)
                        print str(index_words_dict[train_x[doc_id][hobj1]])+"\t"+str(index_words_dict[train_x[doc_id][hobj2]])+"\t"+str(index_words_dict[train_x[doc_id][tobj1]])+"\t"+str(index_words_dict[train_x[doc_id][tobj2]])
                        print str(train_x[doc_id][hobj1])+"\t"+str(train_x[doc_id][hobj2])+"\t"+str(train_x[doc_id][tobj1])+"\t"+str(train_x[doc_id][tobj2])
                        print str(src_offset)+"\t"+str(trg_offset)+"\t"+doc.source[src_offset:src_offset+src_length]+"\t"+doc.source[trg_offset:trg_offset+trg_length]

                        print "1"'''
                    elif label['polarity'] == 'neg' :
                        train_y_labels_doc.append(2)
                        train_y_ids_doc.append((src_id, trg_id, label))
                        '''print str(src_id)+"\t"+str(trg_id)+"\t"+str(hobj1)+"\t"+str(hobj2)+"\t"+str(tobj1)+"\t"+str(tobj2)
                        print str(index_words_dict[train_x[doc_id][hobj1]])+"\t"+str(index_words_dict[train_x[doc_id][hobj2]])+"\t"+str(index_words_dict[train_x[doc_id][tobj1]])+"\t"+str(index_words_dict[train_x[doc_id][tobj2]])
                        print str(src_offset)+"\t"+str(trg_offset)+"\t"+doc.source[src_offset:src_offset+src_length]+"\t"+doc.source[trg_offset:trg_offset+trg_length]

                        print "2"'''
                    elif label['polarity'] == 'none' :
                        train_y_labels_doc.append(0)
                        train_y_ids_doc.append((src_id, trg_id, None))
                        #print "0"
                else :
                    #train_y_labels_doc.append('O')
                    train_y_labels_doc.append(0)
                    train_y_ids_doc.append((src_id, trg_id, None))
                    #print "0--"

            #else : 
            #    print "Duplicate!!!!"+str([hobj1, hobj2, tobj1, tobj2])
        
        if doc_id not in train_y_relations.keys() : 
            train_y_relations[doc_id] = train_y_relations_doc
            train_y_labels[doc_id] = train_y_labels_doc
            train_y_ids[doc_id] = train_y_ids_doc

            
        else : 
            initial = train_y_relations[doc_id]
            initial += train_y_relations_doc
            train_y_relations[doc_id] = initial

            initial_labels = train_y_labels[doc_id]
            initial_labels += train_y_labels_doc
            train_y_labels[doc_id] = initial_labels

            initial_ids = train_y_ids[doc_id]
            intial_ids += train_y_ids_doc
            train_y_ids[doc_id] = initial_ids

        #exit()
        #print "====================================================="

        all_authors[doc.doc_id] = author
        #print author
        #if doc_count > 0 : 
        #    break

    #exit()



    
    #exit()

    #print train_y_relations
    #print train_y_labels
    print "Pickling....."
    
    with open(os.path.join(DATA_ROOT, "sent_Testdata_temp.pkl"), "wb") as f:
        pickle.dump((train_x, train_x_ind, train_y_relations, train_y_labels, train_y_ids), f)
    
    '''print("pickling...")
    with open(os.path.join(DATA_ROOT, "sent_data.pkl"), "wb") as f:
        pickle.dump((X_sent, y_sent, doc_id_sent), f)

    with open(os.path.join(DATA_ROOT, "belief_data.pkl"), "wb") as f:
        pickle.dump((X_belief, y_belief, doc_id_belief), f)'''
