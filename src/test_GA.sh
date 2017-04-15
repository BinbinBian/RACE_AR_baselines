model=model_GA.pkl.gz
gpu=gpu2
echo "!!!test"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python main.py -train_file ../data/data/train -dev_file ../data/data/test -embedding_size 100 -pre_trained ${model} -test_only True -test_output test_GA_all.txt -model GA
echo "!!!dev"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python main.py -train_file ../data/data/train -dev_file ../data/data/dev -embedding_size 100 -pre_trained ${model} -test_only True -model GA
#echo "!!!test/mctest_160"
#THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python main.py -train_file ../data/data/train -dev_file ../data/data/mctest/test/mc160 -embedding_size 100 -pre_trained ${model} -test_only True -model GA
#echo "!!!test/mctest_500"
#THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python main.py -train_file ../data/data/train -dev_file ../data/data/mctest/test/mc500 -embedding_size 100 -pre_trained ${model} -test_only True -model GA
echo "!!!test/chuzhong"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python main.py -train_file ../data/data/train -dev_file ../data/data/test/chuzhong -embedding_size 100 -pre_trained ${model} -test_only True -model GA
echo "!!!test/gaozhong"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python main.py -train_file ../data/data/train -dev_file ../data/data/test/gaozhong -embedding_size 100 -pre_trained ${model} -test_only True -model GA
