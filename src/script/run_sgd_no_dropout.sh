suffix=no_dropout

THEANO_FLAGS="mode=FAST_RUN,device=gpu1,floatX=float32" stdbuf -i0 -e0 -o0 python main.py -train_file ../data/data/train -dev_file ../data/data/dev -embedding_file ../data/embedding/glove.6B.100d.txt -optimizer sgd -model_file model_${suffix}.pkl.gz -dropout_rate 0.0 -log_file log_${suffix}.txt
