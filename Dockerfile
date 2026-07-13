FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt setup.py ${LAMBDA_TASK_ROOT}/
COPY src/ ${LAMBDA_TASK_ROOT}/src/

RUN pip install -r requirements.txt --no-cache-dir

COPY application.py ${LAMBDA_TASK_ROOT}/
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/
COPY templates/ ${LAMBDA_TASK_ROOT}/templates/
COPY data/ ${LAMBDA_TASK_ROOT}/data/

CMD ["lambda_handler.handler"]