# Use a base image provided by AWS for Python 3.9 runtime 
FROM public.ecr.aws/lambda/python:3.9-x86_64

# Set the working directory
WORKDIR /var/task

# Copy requirements.txt
COPY requirements.txt ./

# Copy function code
COPY app.py ./
COPY dynamodb.py ./

# Install the specified packages
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.lambda_handler" ]

