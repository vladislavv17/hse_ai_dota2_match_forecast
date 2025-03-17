# Используем официальный базовый образ AWS Lambda для Python 3.8
FROM public.ecr.aws/lambda/python:3.8

WORKDIR ${LAMBDA_TASK_ROOT}

# Копируем исходный код и requirements.txt в рабочую директорию
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY requirements.txt  .
RUN pip install --no-cache-dir -r requirements.txt

# Точка входа для AWS Lambda
CMD [ "lambda_function.lambda_handler" ]
