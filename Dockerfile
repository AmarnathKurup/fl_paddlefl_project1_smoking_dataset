FROM python:3.7

WORKDIR /workspace

RUN pip install --upgrade pip

RUN pip install protobuf==3.20.3

RUN pip install paddlepaddle==1.8.5

RUN pip install paddle_fl==0.1.1

RUN pip install pandas numpy scikit-learn

CMD ["/bin/bash"]