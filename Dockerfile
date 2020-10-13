FROM debian:sid-20200803-slim as arcise-env

RUN apt-get update
RUN apt-get install -y build-essential unzip wget tar cmake ninja-build gdb git python3
RUN wget https://github.com/llvm/llvm-project/archive/8427885e27813c457dccb011f65e8ded74444e31.zip
RUN unzip 8427885e27813c457dccb011f65e8ded74444e31.zip
RUN mv llvm-project-8427885e27813c457dccb011f65e8ded74444e31 llvm-project
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
RUN apt-get install -y --no-install-recommends clang-format

WORKDIR /
RUN git clone https://github.com/apache/arrow.git -b apache-arrow-1.0.1
RUN mkdir /arrow/cpp/build
WORKDIR /arrow/cpp/build
RUN cmake -DCMAKE_BUILD_TYPE=Debug ..
RUN cmake --build . --target install -j 4

WORKDIR /
RUN useradd -ms /bin/bash docker


