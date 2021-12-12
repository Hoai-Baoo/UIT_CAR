FROM tensorflow/tensorflow:2.7.0-gpu

WORKDIR VIAM-UIT

COPY . /VIAM-UIT

RUN cd /VIAM-UIT && \
    python3 -m pip install --upgrade pip && \
    pip3 install -r requirements.txt

RUN apt update && apt install -y libsm6 libxext6 \
    && apt-get install -y libxrender-dev

CMD python3 Viam_hcmut.py
