FROM python:3.12.6
ENV MY_DIR=/abeec
WORKDIR ${MY_DIR}
COPY . .
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt

CMD bash