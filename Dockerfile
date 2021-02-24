FROM debian:sid-20200803-slim as arcise-env

RUN apt-get update
RUN apt-get install -y build-essential unzip wget tar cmake ninja-build gdb git python3
RUN wget https://github.com/llvm/llvm-project/archive/1267bb2e416e42f9c3bbfa7b6cbf4975fa7aa546.zip
RUN unzip 1267bb2e416e42f9c3bbfa7b6cbf4975fa7aa546.zip
RUN mv llvm-project-1267bb2e416e42f9c3bbfa7b6cbf4975fa7aa546 llvm-project
RUN mkdir llvm-project/build

WORKDIR llvm-project/build

RUN cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_TARGETS_TO_BUILD="X86;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Debug \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_INSTALL_UTILS=ON

RUN cmake --build . --target install -j 2

# installing some development tools here in order to avoid rebuilding dependencies after
# after each tool would be added
RUN apt-get update
RUN apt-get install -y --no-install-recommends clang-format

WORKDIR /
RUN git clone https://github.com/apache/arrow.git -b apache-arrow-1.0.1
RUN mkdir /arrow/cpp/build
WORKDIR /arrow/cpp/build
RUN cmake -DCMAKE_BUILD_TYPE=Debug ..
RUN cmake --build . --target install -j 4

RUN apt-get update && apt-get install -y neovim ccls fzf ripgrep nodejs npm tmux libboost-all-dev

RUN useradd -ms /bin/bash docker
USER docker
CMD /bin/bash
