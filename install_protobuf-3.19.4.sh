#
# build protobuf using cpp implementation
# https://jkjung-avt.github.io/tf-trt-revisited/
#
#!/bin/bash

PROTOBUF_VERSION=3.19.4
PROTOBUF_URL=https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}
PROTOBUF_DIR=protobuf-python-${PROTOBUF_VERSION}
PROTOC_DIR=protoc-${PROTOBUF_VERSION}-linux-aarch_64

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp

# remove previous installation of python3 protobuf module
sudo pip3 uninstall -y protobuf

cd /tmp && \
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${PROTOBUF_URL}/$PROTOBUF_DIR.zip && \
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${PROTOBUF_URL}/$PROTOC_DIR.zip && \
    unzip ${PROTOBUF_DIR}.zip -d ${PROTOBUF_DIR} && \
    unzip ${PROTOC_DIR}.zip -d ${PROTOC_DIR} && \
    cp ${PROTOC_DIR}/bin/protoc /usr/local/bin/protoc && \
    cd ${PROTOBUF_DIR}/protobuf-${PROTOBUF_VERSION} && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local && \
    make -j$(nproc) && \
    make check -j4 && \
    make install && \
    ldconfig && \
    cd python && \
    python3 setup.py build --cpp_implementation && \
    python3 setup.py bdist_wheel --cpp_implementation && \
    cp dist/*.whl /opt && \
    pip3 install dist/*.whl && \
    cd ../../../ && \
    rm ${PROTOBUF_DIR}.zip && \
    rm ${PROTOC_DIR}.zip && \
    rm -rf ${PROTOBUF_DIR} && \
    rm -rf ${PROTOC_DIR}

#RUN python3 setup.py install --cpp_implementation && \
#RUN pip3 install protobuf==${PROTOBUF_VERSION} --install-option="--cpp_implementation" --no-cache-dir --verbose

sudo pip3 show protobuf && \
    protoc --version
