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
sudo -H pip3 uninstall -y protobuf

cd /tmp && \
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${PROTOBUF_URL}/$PROTOBUF_DIR.zip && \
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${PROTOBUF_URL}/$PROTOC_DIR.zip && \
    unzip ${PROTOBUF_DIR}.zip -d ${PROTOBUF_DIR} && \
    unzip ${PROTOC_DIR}.zip -d ${PROTOC_DIR} && \
    sudo cp ${PROTOC_DIR}/bin/protoc /usr/local/bin/protoc && \
    cd ${PROTOBUF_DIR}/protobuf-${PROTOBUF_VERSION} && \
    sudo sed -i '/^TEST_F(IoTest, LargeOutput)/i /*' src/google/protobuf/io/zero_copy_stream_unittest.cc
    sudo sed -i '/^TEST_F(IoTest, FileIo)/i */' src/google/protobuf/io/zero_copy_stream_unittest.cc
    sudo ./autogen.sh && \
    sudo ./configure --prefix=/usr/local && \
    sudo make -j$(nproc) && \
    sudo make check -j4 && \
    sudo make install && \
    sudo ldconfig && \
    cd python && \
    sudo python3 setup.py build --cpp_implementation && \
    sudo python3 setup.py test --cpp_implementation && \
    sudo python3 setup.py bdist_wheel --cpp_implementation && \
    sudo cp dist/*.whl /opt && \
    sudo -H pip3 install dist/*.whl && \
    cd ../../../ && \

sudo -H pip3 show protobuf && \
    protoc --version
