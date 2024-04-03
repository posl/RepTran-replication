FROM python:3.7.16
WORKDIR /src
ADD bash_setting /root/.bashrc
ADD requirements.txt /src
RUN apt-get -qq update && \
		apt-get install -y vim build-essential tree procps
RUN pip install --upgrade pip 
RUN pip install -r requirements.txt
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
		&& apt-get install git-lfs && git lfs install
EXPOSE 8888
EXPOSE 6006