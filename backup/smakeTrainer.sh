/usr/local/cuda-9.0/bin/nvcc  -arch=compute_35 -rdc=true -Xptxas -O3 -g -I . ryggrad/src/base/FileParser.cc ryggrad/src/general/DNAVector.cc ryggrad/src/base/StringUtil.cc ryggrad/src/util/mutil.cc src/analysis/8s.cu ryggrad/src/ml/NNIO.cc ryggrad/src/ml/NNet.cc -o sTrainer -lcublas -lz


