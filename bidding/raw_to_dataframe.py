import numpy as np
import pandas as pd
import datetime
import time
import csv
from tqdm import tqdm
import os
from collections import Counter
import pickle
import sys
import getopt
from ua_parser import user_agent_parser

dtype = {'bid id': str,
         'timestamp': str,
         'log type': str,
         'ipinyou id': str,
         'user-agent': str,
         'ip':str,
         'region id':str,
         'city id':str,
         'ad exchange':str,
         'domain':str,
         'url':str,
         'anonymous url':str,
         'ad slot id':str,
         'ad slot width':float,
         'ad slot height':float,
         'ad slot visibility':str,
         'ad slot format':str,
         'ad slot floor price':float,
         'creative id':str,
         'bidding price':float,
         'paying price':float,
         'landing page url':str,
         'advertiser id':str,
         'user profile ids':str
}

regions = []
with open('./ipinyou.contest.dataset/region.cn.txt','r') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        regions.append(row[0])
cities = []
with open('./ipinyou.contest.dataset/city.cn.txt','r') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        cities.append(row[0])

def preprocess_time(ts):
    ts = str(ts)
    year = int(ts[0:4])
    month = int(ts[4:6])
    day = int(ts[6:8])
    weekday = datetime.date(year, month, day).weekday()
    hour = int(ts[8:10])
    minute = int(ts[10:12])
    sec = int(ts[12:14])
    daysec =  hour * 3600 + minute * 60 + sec
    return weekday, daysec

def preprocess(imps):
    imps = imps.drop_duplicates(subset='bid id',keep='last')
    # remove region and city not seen
    imps = imps[imps['region id'].isin(regions)]
    imps = imps[imps['city id'].isin(cities)]

    # remove user-agent null
    imps = imps[~imps['user-agent'].isnull()]

    # preprocess user agent
    parsed_ua = imps['user-agent'].apply(user_agent_parser.Parse)
    imps['ua os'] = parsed_ua.apply(lambda x: x['os']['family'])
    imps['ua browser'] = parsed_ua.apply(lambda x: x['user_agent']['family'])

    # parse time stamp into two column
    imps['week day'], imps['sec in day'] = zip(*imps['timestamp'].map(preprocess_time))

    # transform numeric data to float
    imps['ad slot width'] = imps['ad slot width'].astype('float32')
    imps['ad slot height'] = imps['ad slot height'].astype('float32')
    imps['ad slot floor price'] = imps['ad slot floor price'].astype('float32')
    imps['paying price'] = imps['paying price'].astype('float32')
    imps['sec in day'] = imps['sec in day'].astype('float32')
    imps['bidding price'] = imps['bidding price'].astype('float32')

    # save to pickle
    imps = imps[imps['paying price'] != 0]
    return imps


if __name__ == '__main__':
    # get parameter in command line
    try:
        myopts, args = getopt.getopt(sys.argv[1:],"i:o:")
    except getopt.GetoptError as e:
        print (str(e))
        print("Usage: %s -i input data root -o output path" % sys.argv[0])
        sys.exit(2)

    root_dir = ''
    output_dir = ''
    for o, a in myopts:
        if o == '-i':
            if a[-1] != '/':
                print('path should end with /')
                sys.exit(2)
            root_dir=a
        elif o == '-o':
            if a[-1] != '/':
                print('path should end with /')
                sys.exit(2)
            output_dir=a

    os.mkdir(output_dir)
    # load input data
    print('loading data...')
    file_names = os.listdir(root_dir)
    imp_file_names = sorted([f for f in file_names if f[:3]=='imp'])
    l = []
    for fname in tqdm(imp_file_names):
        # date = fname[3:-4]
        imps = pd.read_csv(root_dir+fname,dtype=dtype)
        imps = preprocess(imps)
        imps.to_csv(output_dir+fname, index=False)
        l.append(imps)
    bids =  pd.concat(l)
    bids.to_csv(root_dir+'allbid.csv', index=False)

