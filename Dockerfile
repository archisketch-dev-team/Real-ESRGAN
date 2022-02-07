FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN apt-get update
RUN apt-get -y install libglib2.0-0
RUN apt-get -y install libgl1-mesa-glx
RUN pip install basicsr
RUN pip install --no-cache-dir -r requirements.txt


COPY . .

RUN python setup.py develop

# ENTRYPOINT ["python"]

# CMD [ "application.py" ]