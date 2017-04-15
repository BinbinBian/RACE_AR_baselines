suffix=fix_emb
THEANO_FLAGS="mode=FAST_RUN,device=gpu0,floatX=float32" stdbuf -i0 -e0 -o0 python main.py -train_file ../data/data/train -dev_file ../data/data/dev -embedding_file ../data/embedding/glove.6B.100d.txt -optimizer sgd -tune_embedding False -model_file model_${suffix}.pkl.gz | tee log_${suffix}.txt
