from easydict import EasyDict
import yaml
import logging


def build_easydict_nodecls():
    MDT = EasyDict()
    MDT.MODEL = EasyDict()
    MDT.MODEL.NAME = 'STBALE'
    MDT.MODEL.PARAM = {
        'threshold':1,  #help='threshold')
        'jt':0.03,  #help='jaccard threshold')
        'cos':0.25, # help='cosine similarity threshold')
        'k':7 ,  #help='add k neighbors')
        'alpha':0.6,  #help='add k neighbors')
        'beta':2, # help='the weight of selfloop')
    }
    return MDT