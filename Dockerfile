# app/Dockerfile

FROM python:3.9.6

CMD mkdir /app

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip3 install --upgrade pip setuptools wheel      

RUN pip3 install torch==1.8.0 torchvision==0.9.0  -f https://download.pytorch.org/whl/cu101/torch_stable.html
RUN pip3 install --no-cache-dir streamlit
RUN pip3 install -r requirements.txt

EXPOSE 8501

COPY . /app

ENTRYPOINT ["streamlit", "run"]

CMD ["objdet.py"]