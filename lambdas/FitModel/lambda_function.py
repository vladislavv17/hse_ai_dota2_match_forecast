import json
import base64
import pickle
from io import StringIO

import boto3
import pandas as pd

# Инициализируем клиент S3
s3_client = boto3.client('s3')


def lambda_handler(event, _context):
    """
    Ожидается, что event содержит следующие ключи:
      - "pickle_file": base64-кодированная строка с pickle-моделью,
      - "csv_data": base64-кодированная строка с CSV-данными,
      - "params": словарь, содержащий:
          - параметры для метода fit модели,
          - "model_name": название модели,
          - "s3_bucket": имя S3 bucket,
          - "s3_key": путь (ключ) для сохранения модели в bucket.
    """
    try:  # pylint: disable=broad-exception-caught
        # Извлечение входных данных
        if 'body' in event:
            event = json.loads(event['body'])

        # Извлечение входных данных
        pickle_file_b64 = event.get('pickle_file')
        csv_data_b64 = event.get('csv_data')
        params = event.get('params', {})

        if not pickle_file_b64 or not csv_data_b64:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Не переданы все необходимые файлы: pickle_file и csv_data'
                })
            }

        # Получаем из params данные для S3 и название модели
        model_name, s3_bucket, s3_key = (
            params.get('model_name'),
            params.get('s3_bucket'),
            params.get('s3_key')
        )
        if not model_name or not s3_bucket or not s3_key:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'В params должны быть указаны model_name, s3_bucket и s3_key'
                })
            }

        # Декодирование и загрузка данных
        model = pickle.loads(base64.b64decode(pickle_file_b64))
        df = pd.read_csv(StringIO(base64.b64decode(csv_data_b64).decode('utf-8')))
        if 'target' not in df.columns:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'В CSV не найден столбец "target".'})
            }

        # Подготовка данных: X - признаки, y - целевая переменная
        y = df['target']
        X = df.drop(columns=['target'])

        # Обучение модели с дополнительными параметрами (если они переданы)
        model.fit(X, y, **params)

        # Сериализация обученной модели и сохранение в S3
        fitted_model_pickle = pickle.dumps(model)
        fitted_model_b64 = base64.b64encode(fitted_model_pickle).decode('utf-8')
        s3_client.put_object(Bucket=s3_bucket, Key=s3_key, Body=fitted_model_pickle)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Модель успешно обучена и сохранена в S3',
                'model_name': model_name,
                's3_bucket': s3_bucket,
                's3_key': s3_key,
                'fitted_model': fitted_model_b64
            })
        }
    except Exception as e:  # pylint: disable=broad-exception-caught
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
