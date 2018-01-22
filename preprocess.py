#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jyao

"""
from __future__ import print_function
from urllib.request import urlopen
import sys
import os
import glob
import zipfile

import utils
import options
import data.ddi2013 as ddi2013

def download(url, dirpath):
    filename = url.split('/')[-1]
    filepath = os.path.join(dirpath, filename)
    if os.path.exists(os.path.join(dirpath, filename)):
        return filepath
    try:
        u = urlopen(url)
    except:
        print("URL %s failed to open" %url)
        raise Exception
    try:
        f = open(filepath, 'wb')
    except:
        print("Cannot write %s" %filepath)
        raise Exception
    try:
        filesize = int(u.info().get("Content-Length"))
    except:
        print("URL %s failed to report length" %url)
        raise Exception
    print("Downloading: %s Bytes: %s" % (filename, filesize))

    downloaded = 0
    block_sz = 8192
    status_width = 70
    while True:
        buf = u.read(block_sz)
        if not buf:
            print('')
            break
        else:
            print('', end='\r')
        downloaded += len(buf)
        f.write(buf)
        status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") %
            ('=' * int(float(downloaded) / filesize * status_width) + '>', downloaded * 100. / filesize))
        print(status, end='')
        sys.stdout.flush()
    f.close()
    return filepath

def unzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)

def download_tagger(dirpath):
    tagger_dir = 'stanford-tagger'
    if os.path.exists(os.path.join(dirpath, tagger_dir)):
        print('Found Stanford POS Tagger - skip')
        return
#    url = 'https://nlp.stanford.edu/software/stanford-postagger-2017-06-09.zip'
    url = 'http://nlp.stanford.edu/software/stanford-postagger-2015-01-29.zip'
    filepath = download(url, dirpath)
    zip_dir = ''
    with zipfile.ZipFile(filepath) as zf:
        zip_dir = zf.namelist()[0]
        zf.extractall(dirpath)
    os.remove(filepath)
    os.rename(os.path.join(dirpath, zip_dir), os.path.join(dirpath, tagger_dir))

def download_parser(dirpath):
    parser_dir = 'stanford-parser'
    if os.path.exists(os.path.join(dirpath, parser_dir)):
        print('Found Stanford Parser - skip')
        return
#    url = 'https://nlp.stanford.edu/software/stanford-parser-full-2017-06-09.zip'
    url = 'http://nlp.stanford.edu/software/stanford-parser-full-2015-01-29.zip'
    filepath = download(url, dirpath)
    zip_dir = ''
    with zipfile.ZipFile(filepath) as zf:
        zip_dir = zf.namelist()[0]
        zf.extractall(dirpath)
    os.remove(filepath)
    os.rename(os.path.join(dirpath, zip_dir), os.path.join(dirpath, parser_dir))

def dependency_parse(filepath, cp='', tokenize=True):
    print('\nDependency parsing ' + filepath)
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.toks')
    parentpath = os.path.join(dirpath, filepre + '.parents')
    relpath =  os.path.join(dirpath, filepre + '.rels')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s DependencyParse -tokpath %s -parentpath %s -relpath %s %s < %s'
        % (cp, tokpath, parentpath, relpath, tokenize_flag, filepath))
    os.system(cmd)

def constituency_parse(filepath, cp='', tokenize=True):
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.toks')
    parentpath = os.path.join(dirpath, filepre + '.cparents')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s ConstituencyParse -tokpath %s -parentpath %s %s < %s'
        % (cp, tokpath, parentpath, tokenize_flag, filepath))
    os.system(cmd)

def build_vocab(filepaths, dst_path, lowercase=True):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')

def split(filepath, dst_dir, delim="|"):
    with open(filepath) as datafile, \
         open(os.path.join(dst_dir, 'sent.txt'), 'w') as sentfile, \
         open(os.path.join(dst_dir, 'other.txt'), 'w') as otherfile:
            for line in datafile:
                sent_id, pair_id, e1, e2, ddi, dtype, sent, p1, p2 = line.strip().split(delim)
                sentfile.write(sent + '\n')
                otherfile.write(delim.join([sent_id, pair_id, e1, e2, ddi, dtype, p1, p2]) + '\n')

def parse(dirpath, cp='', dep=True, const=True):
    if dep:
        dependency_parse(os.path.join(dirpath, 'sent.txt'), cp=cp, tokenize=True)
    if const:
        constituency_parse(os.path.join(dirpath, 'sent.txt'), cp=cp, tokenize=True)
    
def main():
    # build parser
    parser = options.get_parser('Preprocessor')
    options.add_dataset_args(parser)
    options.add_preprocessing_args(parser)
    args = parser.parse_args()
    print(args)
    
    # make dirs
    base_dir = os.path.dirname(os.path.realpath(__file__))
    lib_dir = os.path.join(base_dir, 'lib')
    processed_dir = args.processed_dir
    train_dir = os.path.join(processed_dir, 'train')
    val_dir = os.path.join(processed_dir, 'val')
    test_dir = os.path.join(processed_dir, 'test')
    utils.make_dirs([args.processed_dir, lib_dir, train_dir, val_dir, test_dir])
    
    # preprocess
    train_corpus = ddi2013.preprocess_ddi(os.path.join(args.raw_dir, 'train'), position=True)
    test_corpus = ddi2013.preprocess_ddi(os.path.join(args.raw_dir, 'test'), position=True)
    
    # get train targets
    input_targets = utils.map_iterable([item.type for item in train_corpus], ddi2013.target_map)
    
    # train/val split
    train_corpus, _, val_corpus, _ = utils.stratified_shuffle_split(train_corpus, input_targets, train_size=args.train_size)
    
    # write to files
    if not os.path.isdir(args.processed_dir):
        os.mkdir(args.processed_dir)
    ddi2013.write_to_file(train_corpus, os.path.join(args.processed_dir, 'train.ddi'))
    ddi2013.write_to_file(val_corpus, os.path.join(args.processed_dir, 'val.ddi'))
    ddi2013.write_to_file(test_corpus, os.path.join(args.processed_dir, 'test.ddi'))

    # download necessary tools
    download_tagger(lib_dir)
    download_parser(lib_dir)
    
    # parse
    #TODO: sometimes compile failed
    os.system('CLASSPATH="lib:lib/stanford-parser/stanford-parser.jar:lib/stanford-parser/stanford-parser-3.5.1-models.jar"')
    os.system('javac -cp $CLASSPATH lib/*.java')
   
    print('=' * 80)
    print('Preprocessing dataset')
    print('=' * 80)

    # java classpath for calling Stanford parser
    classpath = ':'.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.5.1-models.jar')])

    # split into separate files
    split(os.path.join(processed_dir, 'train.ddi'), train_dir)
    split(os.path.join(processed_dir, 'val.ddi'), val_dir)
    split(os.path.join(processed_dir, 'test.ddi'), test_dir)

    # parse sentences
    for d in [train_dir, val_dir, test_dir]:
        parse(d, cp=classpath, dep=args.dep, const=args.const)

    # get vocabulary
    build_vocab(
        glob.glob(os.path.join(processed_dir, '*/*.toks')),
        os.path.join(processed_dir, 'vocab.txt'))
    build_vocab(
        glob.glob(os.path.join(processed_dir, '*/*.toks')),
        os.path.join(processed_dir, 'vocab-cased.txt'),
        lowercase=False)
    
if __name__ == '__main__':
    main()
    
    