from itertools import product
from lxml import etree as et
from loaders import extract_mentions, extract_sent, extract_belief, extract_mentions_pred
from loaders import *
from collections import Counter
from loaders import iter_best_split



#allpairs_list = []


def ret_mapping(doc_id,sid,tid):
    mapping = {}
    x = doc_id + ".cmp.txt.conll8.parsed"           
    token_num = 0
    token_id = 0
    sent_id = 0
    with open("parsed/"+x,"r") as f:
        for line in f:                   
            current = line.split("\t")  
            #print current                
            if len(current)  != 1:                      
                token_num = token_num + 1
                # output.write(str(token_num)+" "+current[1]+"      "),
                # output.write(str(sent_id)+"       "+str(token_id))
                mapping[(sent_id,token_id)] = token_num
                token_id = token_id + 1
                # for count in range(1,len(current)-1):
                #   output.write(current[count]+"   "),
                #output.write(current[len(current)-1]),
                # output.write("\n")
            else:
                # output.write(line)
                token_id = 0
                sent_id = sent_id + 1
    #print mapping
    return mapping[(sid,tid)]

def pairs_from_doc(best_file, strict_neg=False):
    _mentions = extract_mentions(best_file.ere)
    entity_mentions, relation_mentions, event_mentions = _mentions
    entity_mention_keys = sorted(list(entity_mentions.keys()))
    relation_mention_keys = sorted(list(relation_mentions.keys()))
    event_mention_keys = sorted(list(event_mentions.keys()))
    sentiments = best_file.annotation.find('sentiment_annotations')

    st_ent = sentiments.find('entities').getchildren()
    st_rel = sentiments.find('relations').getchildren()
    st_evt = sentiments.find('events').getchildren()

    
    sent_links = {}

    
    sent_obj_links = set()
    sent_offset_links = set()

    obj_id = {k: v.attrib['id'] for k, v in best_file.objects.items()}
    obj_id[None] = "None"
    # for x in best_file.objects.items():
    #     print x[1].attrib     #{'type': 'PER', 'id': 'ent-8', 'specificity': 'specific'}
    #obj_offset = {k: v.attrib['offset'] for k, v in best_file.objects.items()}
    #obj_offset[None] = -1

    for link in extract_sent(st_ent + st_rel + st_evt):
        src = link.get('src')  # None if missing
        trg = link['trg']
        sent_links[src, trg] = {
            k: v for k, v in link.items()
            if k in ['polarity', 'sarcasm']
            
        }
        # for k, v in link.items():
        #     if k in ['polarity', 'sarcasm']:
                #print str(src) + "   " + str(trg)+"   " + str(v)

        if link['polarity'] != 'none':
            sent_obj_links.add((obj_id[src], obj_id[trg]))
            #sent_offset_links.add((obj_offset[src], obj_offset[trg]))

    sent_candidates = product([None] + entity_mention_keys,
                              entity_mention_keys +
                              relation_mention_keys +
                              event_mention_keys)

    sent_pairs = []
    offset_pairs = []
    belief_pairs = []

    for src, trg in sent_candidates:
        if src == trg:
            continue

        y = sent_links.get((src, trg))
        #print "HELLO"+" "+str(src) + "   " + str(trg)+"   " + str(y)

        if strict_neg and (y is None or y['polarity'] == 'none'):
            if (obj_id[src], obj_id[trg]) in sent_obj_links:
                continue

        sent_pairs.append((src, trg, y))
        #offset_pairs.append((obj_offset[src],obj_offset[trg]))

   

    return sent_pairs #, offset_pairs


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

    
    obj_id = {k: v.attrib['id'] for k, v in best_file.objects.items()}
    obj_id[None] = "None"

    sent_pairs = []
    belief_pairs = []

    sent_candidates = product([None] + entity_mention_keys,
                              entity_mention_keys +
                              relation_mention_keys +
                              event_mention_keys)

    


    for src, trg in sent_candidates:
        
        if src == trg:
            continue

        sent_pairs.append((src, trg, -1))

    return sent_pairs  


def xml_from_pairs(sent_pairs, belief_pairs):
    doc = et.Element("committed_belief_doc")
    
    sentiments = et.SubElement(doc, "sentiment_annotations")
    sent_ent = et.SubElement(sentiments, "entities")
    sent_rel = et.SubElement(sentiments, "relations")
    sent_evt = et.SubElement(sentiments, "events")

    name = dict(relm="relation", em="event", m="entity")
    sent_parent = dict(relm=sent_rel, em=sent_evt, m=sent_ent)
    

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

    DATA_ROOT = '/home/anusha/Desktop/6740 Project/CURRENT/data/'
    for doc in iter_best_split(DATA_ROOT, 'train'):        
        sent_pairs = pairs_from_doc(doc, strict_neg=True)
        out = doc.doc_id + "_pairlist.txt"
        output = open(os.path.join("pairs/",out),"w+")
        #output2 = open(os.path.join("pairs_y/",out),"w+")
        #print(len(sent_pairs))
        #print(sent_pairs)
        #print(Counter(y["polarity"] if y is not None else "none" for _, _, y in sent_pairs))
        #i = 0
        offset_1 = -1
        offset_2 = -1
        length1 = -1
        length2 = -1
        start = [[-1,-1],[-1,-1]]
        end = [[-1,-1],[-1,-1]]
        start1 = None
        start2 = None
        end1 = None
        end2 = None
        y = None
        allpairs_list = []
        all_pairs = {}
        #print doc.doc_id
        for sent_pair in sent_pairs:
            m_id1 = sent_pair[0]
            m_id2 = sent_pair[1] 
            y = sent_pair[2]                   
            
            if m_id1!=None:          
                mention1 = doc.mentions[m_id1]  #TypeError: 'dict' object is not callable
                flag = 10
                cnt =0
                while cnt<flag:
                    try:
                        offset_1 = int(mention1.attrib['offset'])
                        length1 = int(mention1.attrib['length'])
                        cnt = 11
                    except:
                        cnt = cnt+1
                start = doc.offset_to_tokens(offset_1, length1)
                if start!=(None,None):
                    # start1 = start[0][1]
                    # end1 = start[1][1]
                    start1 = ret_mapping(doc.doc_id,start[0][0],start[0][1])
                    end1 = ret_mapping(doc.doc_id,start[1][0],start[1][1])

                # print m_id1
                # print offset_1
                # print length1
                # print start1
                # print end1
                # print "*********"
            
            if m_id2!=None:
                mention2 = doc.mentions[m_id2]  #TypeError: 'dict' object is not callable
                flag = 10
                cnt = 0
                while cnt<flag:
                    try:
                        offset_2 = int(mention2.attrib['offset'])
                        length2 = int(mention2.attrib['length'])                        
                        cnt = 11
                    except:
                        cnt = cnt + 1
                end = doc.offset_to_tokens(offset_2, length2)
                #print end
                if end!=(None,None):
                    # start2 = end[0][1]
                    # end2 = end[1][1]
                    start2 = ret_mapping(doc.doc_id,end[0][0],end[0][1])
                    end2 = ret_mapping(doc.doc_id,end[1][0],end[1][1])
            
            #i = i + 1
            
            # if sent_pair[0]=='m-366' or sent_pair[0]=='m-360':
            #     print sent_pair[0]
            #     print sent_pair[1]
            #     print start
            #     print end
            if (sent_pair[0],sent_pair[1]) not in all_pairs:
                all_pairs[ (sent_pair[0],sent_pair[1])  ] = (sent_pair[2] , offset_1, offset_2,length1,length2,start1,end1,start2,end2)
                #if sent_pair[0]=='m-366' and sent_pair[1]=='m-246':
                    #print "found"
                    #print str(start1)+" "+str(end1)+" "+str(start2)+" "+str(end2)

                if ([start1,end1],[start2,end2]) not in allpairs_list:
                    # if sent_pair[0]=='m-366' and sent_pair[1]=='m-246':
                    #     print "found2"
                    allpairs_list.append(([start1,end1],[start2,end2])) #[start1,end1],[start2,end2]
                    output.write(str(start1)+"  "+str(end1)+"   "+str(start2)+"     "+str(end2)+"   "+str(y)) #+"  "+str(sent_pair[0])+"    "+str(sent_pair[1]))
                    output.write("\n")
                    # output2.write(str(y))
                    # output2.write("\n")
        #print allpairs_list
        
    #print all_pairs



