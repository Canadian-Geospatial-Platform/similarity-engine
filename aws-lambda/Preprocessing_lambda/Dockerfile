# Use a base image provided by AWS for Python 3.9 runtime 
FROM public.ecr.aws/lambda/python:3.9-x86_64

# Set the working directory
WORKDIR /var/task

# Copy requirements.txt
COPY requirements.txt ./

# Copy function code
COPY app.py ./

# Install the specified packages
RUN pip install -r requirements.txt

# Download the NLTK 'punkt' tokenizer and 'stopwords' data
RUN python -m nltk.downloader -d /var/task/nltk_data punkt stopwords

# Set the NLTK_DATA environment variable to use the included data
ENV NLTK_DATA=/var/task/nltk_data

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.lambda_handler" ]

