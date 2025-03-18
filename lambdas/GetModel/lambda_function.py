import base64
import json

import boto3

# Настройте имя бакета и префикс (папку) в S3
BUCKET_NAME = 'dmyachin-new-models'
PREFIX = 'models'  # Например, 'pickles/'

s3 = boto3.client('s3')


def lambda_handler(_event, _context):
    """
    Return all models that collected in bucket 'dmyachin-new-models'
    in folder 'models'
    """
    # Получаем список объектов по префиксу
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)
    files_data = []

    if 'Contents' in response:
        for obj in response['Contents']:
            key = obj['Key']
            # Обрабатываем только файлы с расширением .pkl
            if key.endswith('.pkl'):
                s3_object = s3.get_object(Bucket=BUCKET_NAME, Key=key)
                file_content = s3_object['Body'].read()
                # Кодируем бинарное содержимое файла в base64,
                # чтобы можно было передать его в JSON
                encoded_data = base64.b64encode(file_content).decode('utf-8')
                files_data.append({
                    'key': key,
                    'data': encoded_data
                })

    return {
        'statusCode': 200,
        'body': json.dumps(files_data)
    }
