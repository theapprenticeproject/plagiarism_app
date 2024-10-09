import pika
import json
import frappe
from .plagiarism_detector import process_image_submission
from frappe.utils.background_jobs import start_worker

def connect_to_rabbitmq():
    rabbitmq_settings = frappe.get_single("RabbitMQ Settings")
    credentials = pika.PlainCredentials(rabbitmq_settings.username, rabbitmq_settings.password)
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host=rabbitmq_settings.host,
        port=rabbitmq_settings.port,
        virtual_host=rabbitmq_settings.virtual_host,
        credentials=credentials))
    return connection

def callback(ch, method, properties, body):
    submission_data = json.loads(body)
    process_image_submission(submission_data)

def start_consuming():
    connection = connect_to_rabbitmq()
    channel = connection.channel()
    rabbitmq_settings = frappe.get_single("RabbitMQ Settings")
    channel.queue_declare(queue=rabbitmq_settings.submission_queue)
    channel.basic_consume(queue=rabbitmq_settings.submission_queue, on_message_callback=callback, auto_ack=True)
    try:
        print("RabbitMQ consumer started. Waiting for messages...")
        channel.start_consuming()
    except KeyboardInterrupt:
        print("RabbitMQ consumer stopped.")
        channel.stop_consuming()
    except Exception as e:
        print(f"Error occurred while consuming messages: {str(e)}")
    finally:
        connection.close()

def start_consumer_background_job():
    while True:
        try:
            start_consuming()
        except Exception as e:
            print(f"Error occurred in the background job: {str(e)}")
            frappe.log_error(f"Error occurred in the background job: {str(e)}", "RabbitMQ Consumer Error")
        finally:
            frappe.db.commit()
            frappe.destroy()

def enqueue_consumer_background_job():
    frappe.enqueue(
        start_consumer_background_job,
        queue="long",
        timeout=None,
        now=frappe.conf.developer_mode or frappe.flags.in_test
    )

if __name__ == "__main__":
    start_worker(queue="long", quiet=True)
    enqueue_consumer_background_job()
