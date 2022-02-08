FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN apt-get update
RUN apt-get -y install libglib2.0-0 wget
RUN apt-get -y install libgl1-mesa-glx
RUN pip install basicsr
RUN pip install --no-cache-dir -r requirements.txt


COPY . .

RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
RUN python setup.py develop

# ENTRYPOINT ["python"]

# CMD [ "application.py" ]