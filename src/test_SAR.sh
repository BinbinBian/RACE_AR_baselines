gpu=gpu0
output_suffix=SAR
option_suffix=''
model=model_${output_suffix}.pkl.gz
echo ${output_suffix}
echo "!!!test"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python main.py -train_file ../data/data/train -dev_file ../data/data/test -embedding_size 100 -pre_trained ${model} -test_only True #-test_output test_${output_suffix}.txt ${option_suffix}
echo "!!!dev"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python main.py -train_file ../data/data/train -dev_file ../data/data/dev -embedding_size 100 -pre_trained ${model} -test_only True ${option_suffix}
echo "!!!test/middle"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python main.py -train_file ../data/data/train -dev_file ../data/data/test/middle -embedding_size 100  -pre_trained ${model} -test_only True ${option_suffix}
echo "!!!test/high"
THEANO_FLAGS="mode=FAST_RUN,device=${gpu},floatX=float32" stdbuf -i0 -e0 -o0 python main.py -train_file ../data/data/train -dev_file ../data/data/test/high -embedding_size 100 -pre_trained ${model} -test_only True ${option_suffix}
