from ubuntu:20.04 AS prepare_deps

ENV DEBIAN_FRONTEND=noninteractive
ENV GIT_SSL_NO_VERIFY=1

RUN apt-get update
RUN apt-get install -y build-essential unzip wget tar cmake ninja-build gdb git python3 libboost-all-dev clang clang-format
RUN wget https://github.com/llvm/llvm-project/archive/refs/tags/llvmorg-12.0.0-rc3.zip
RUN unzip llvmorg-12.0.0-rc3.zip
RUN mv llvm-project-llvmorg-12.0.0-rc3 llvm-project
RUN mkdir llvm-project/build

WORKDIR llvm-project/build

RUN cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Debug \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_INSTALL_UTILS=ON

RUN cmake --build . --target install -j 2


RUN useradd -ms /bin/bash docker
USER docker
CMD /bin/bash
