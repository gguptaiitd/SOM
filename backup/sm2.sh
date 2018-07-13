rm -rf s2
/usr/local/cuda-9.0/bin/nvcc  --gpu-architecture=compute_61 --gpu-code=sm_61 -rdc=true -Xptxas -O3 -g -I . ryggrad/src/base/FileParser.cc ryggrad/src/general/DNAVector.cc ryggrad/src/base/StringUtil.cc ryggrad/src/util/mutil.cc src/analysis/7is.cu ryggrad/src/ml/NNIO.cc ryggrad/src/ml/NNet.cc -o s2 -lcublas_static -lculibos  -lz


