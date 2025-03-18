import json

import boto3
from botocore.exceptions import ClientError

# Инициализация клиента для работы с S3 и ресурса для DynamoDB
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Имя таблицы DynamoDB, которая служит key-value хранилищем
TABLE_NAME = 'MyKeyValueTable'
table = dynamodb.Table(TABLE_NAME)


def lambda_handler(event, context):
    # Ожидаем, что вызов происходит через API Gateway и данные передаются в теле запроса
    try:
        body = event.get('body')
        # Если тело запроса в виде строки, парсим его как JSON
        if isinstance(body, str):
            body = json.loads(body)
    except Exception:  # pylint: disable=broad-exception-caught
        return {
            'statusCode': 400,
            'body': json.dumps('Неверный формат тела запроса')
        }

    # Извлекаем необходимые параметры
    bucket = body.get('bucket')
    directory = body.get('directory')
    file_name = body.get('file_name')

    if not (bucket and directory and file_name):
        return {
            'statusCode': 400,
            'body': json.dumps('Отсутствуют необходимые параметры: bucket, directory, file_name')
        }

    # Формируем ключ S3 (путь к файлу)
    s3_key = f"{directory}/{file_name}"

    # Проверяем наличие файла в S3
    try:
        s3_client.head_object(Bucket=bucket, Key=s3_key)
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            return {
                'statusCode': 404,
                'body': json.dumps('Файл не найден')
            }
        else:
            return {
                'statusCode': 500,
                'body': json.dumps(f'Ошибка при проверке файла: {e}')
            }

    # Если файл найден, устанавливаем значение для записи в DynamoDB (например, имя файла)
    dynamodb_value = file_name

    # Записываем элемент в DynamoDB. В данном случае используется ключ 'id' с уникальным значением.
    try:
        table.put_item(
            Item={
                'id': 'unique_key',      # Замените 'unique_key' на значение, соответствующее логике вашего приложения
                'value': dynamodb_value
            }
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        return {
            'statusCode': 500,
            'body': json.dumps(f'Ошибка при записи в DynamoDB: {e}')
        }

    return {
        'statusCode': 200,
        'body': json.dumps(f'Успех: значение "{dynamodb_value}" успешно записано в DynamoDB')
    }
