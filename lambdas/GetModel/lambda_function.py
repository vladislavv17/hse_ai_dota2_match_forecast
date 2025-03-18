import base64
import json

import boto3

# Настройте имя бакета и префикс (папку) в S3
BUCKET_NAME = 'dmyachin-new-models'
PREFIX = 'models'  # Например, 'pickles/'

s3 = boto3.client('s3')


def lambda_handler(event, context):
    # Получаем список объектов в заданном префиксе
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=PREFIX)
    models_list = []

    if 'Contents' in response:
        for obj in response['Contents']:
            key = obj['Key']
            # Обрабатываем только файлы с расширением .pkl
            if key.endswith('.pkl'):
                s3_object = s3.get_object(Bucket=BUCKET_NAME, Key=key)
                file_content = s3_object['Body'].read()
                try:
                    # Загружаем данные из pickle файла
                    model_data = pickle.loads(file_content)

                    # Формируем словарь с необходимыми полями
                    model_info = {
                        's3_key': key,
                        'model': str(model_data.get('model')),  # преобразуем model в строку, чтобы избежать проблем с сериализацией
                        'params': model_data.get('params'),
                        'classification_report': model_data.get('classification_report'),
                        'learning_curve': model_data.get('learning_curve'),
                        'experiment_id': model_data.get('experiment_id')
                    }

                    models_list.append(model_info)
                except Exception as e:
                    # Можно добавить логирование ошибок или пропустить некорректные файлы
                    print(f"Ошибка при обработке файла {key}: {e}")

    return {
        'statusCode': 200,
        'body': json.dumps(models_list)
    }