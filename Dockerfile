#Dockerfile
FROM python:3.7
RUN mkdir /application
RUN mkdir /application/Scripts
WORKDIR "/application/Scripts"

# Upgrade pip
RUN pip install --upgrade pip

# Update
RUN apt-get update \
    && apt-get clean; rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /usr/share/doc/*

# ADD project files
ADD requirements.txt /application/
ADD Data /application/Data
ADD Scripts /application/Scripts/

RUN pip install -r /application/requirements.txt

#ENTRYPOINT [ "python" ]