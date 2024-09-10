import pika
import json
import frappe
from .plagiarism_detector import process_image_submission

# RabbitMQ connection settings
rabbitmq_config = {
    'host': 'armadillo.rmq.cloudamqp.com',
    'port': 5672,
    'virtual_host': 'fzdqidte',
    'username': 'fzdqidte',
    'password': '0SMrDogBVcWUcu9brWwp2QhET_kArl59',
    'queue': 'submission_queue'
}

# Connect to RabbitMQ
def connect_to_rabbitmq():
    credentials = pika.PlainCredentials(rabbitmq_config['username'], rabbitmq_config['password'])
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host=rabbitmq_config['host'],
        port=rabbitmq_config['port'],
        virtual_host=rabbitmq_config['virtual_host'],
        credentials=credentials))
    return connection

# Callback function to handle incoming messages
def callback(ch, method, properties, body):
    submission_data = json.loads(body)
    process_image_submission(submission_data)

# Function to start consuming RabbitMQ messages
def start_consuming():
    connection = connect_to_rabbitmq()
    channel = connection.channel()
    channel.queue_declare(queue=rabbitmq_config['queue'])

    channel.basic_consume(queue=rabbitmq_config['queue'], on_message_callback=callback, auto_ack=True)
    channel.start_consuming()

# To test or run manually, call start_consuming()
if __name__ == "__main__":
    start_consuming()
