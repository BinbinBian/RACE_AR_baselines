suffix=GA_adam_0.5_lr_0.01

THEANO_FLAGS="mode=FAST_RUN,device=gpu1,floatX=float32" stdbuf -i0 -e0 -o0 python main.py -train_file ../data/data/train -dev_file ../data/data/dev -embedding_file ../data/embedding/glove.6B.100d.txt -optimizer sgd -model_file model_${suffix}.pkl.gz -model GA -num_GA_layers 3 -lr 0.01 -dropout_rate 0.5 -log_file log_${suffix}.txt -optimizer adam
