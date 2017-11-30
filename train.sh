python train.py --train_corpus_path ../../data/drugddi2013/re/train --test_corpus_path ../../data/drugddi2013/re/test --train_path ../../data/drugddi2013/re/train.ddi --test_path ../../data/drugddi2013/re/test.ddi --checkpoint ./checkpoint/re_we_200_100 --load_checkpoint "" --emb_file ../../data/word_embedding/w2v/wikipedia-pubmed-and-PMC-w2v.txt --train_size 0.8 --caseless --embedding_dim 200 --hidden_dim 100 --dropout_ratio 0.5 --lr 0.025 --lr_decay 0.005
