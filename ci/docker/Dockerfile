# Following some suggestions in https://meteoswiss.atlassian.net/wiki/spaces/APN/pages/719684202/Clariden+Alps+environment+setup

#FROM ubuntu:22.04 as builder
FROM nvcr.io/nvidia/pytorch:25.01-py3



# setup
#RUN apt-get update && apt-get install python3-pip python3-venv -y
RUN pip install --upgrade \
        pip
        #ninja
        #wheel
        #packaging
        #setuptools

# update flash-attn
#RUN MAX_JOBS=16 pip install --upgrade --no-build-isolation \
#        flash-attn==2.7.4.post1 -v

# install the rest of dependencies
# TODO: Factor pydeps into a separate file(s)
# TODO: Add versions for things
RUN pip install \
        anemoi-datasets \
        cartopy 
        #matplotlib \
        #numpy \
        #pandas \
        #scipy \
        #torch


# replace pynvml with nvidia-ml-py
#RUN pip uninstall -y pynvml && pip install nvidia-ml-py

#CMD ["python3.11" "src/input_data/interpolate_basic_test.py"]


COPY . /src
WORKDIR /src

