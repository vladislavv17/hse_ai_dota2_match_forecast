import json
import base64
import pickle
import boto3
import pandas as pd
from io import StringIO

# Инициализируем клиент S3
s3_client = boto3.client('s3')


def lambda_handler(event, context):
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
    try:
        # Извлечение входных данных
        pickle_file_b64 = event.get('pickle_file')
        csv_data_b64 = event.get('csv_data')
        params = event.get('params', {})  # включает параметры для fit и данные для S3

        # Проверка обязательных файлов
        if not pickle_file_b64 or not csv_data_b64:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Не переданы все необходимые файлы: pickle_file и csv_data'})
            }

        # Получаем из params данные для S3 и название модели
        model_name = params.get('model_name')
        s3_bucket = params.get('s3_bucket')
        s3_key = params.get('s3_key')

        if not model_name or not s3_bucket or not s3_key:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'В params должны быть указаны model_name, s3_bucket и s3_key'})
            }

        # Декодирование base64
        pickle_bytes = base64.b64decode(pickle_file_b64)
        csv_bytes = base64.b64decode(csv_data_b64)

        # Загрузка модели из pickle
        model = pickle.loads(pickle_bytes)

        # Чтение CSV-данных
        csv_str = csv_bytes.decode('utf-8')
        df = pd.read_csv(StringIO(csv_str))

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

        # Сериализация обученной модели в pickle
        fitted_model_pickle = pickle.dumps(model)
        fitted_model_b64 = base64.b64encode(fitted_model_pickle).decode('utf-8')

        # Сохранение модели в S3 по указанному пути
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
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
