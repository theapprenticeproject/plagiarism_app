import pika
import json
import frappe
from .plagiarism_detector import process_image_submission


def connect_to_rabbitmq():
    """Establish RabbitMQ connection with proper settings"""
    rabbitmq_settings = frappe.get_single("RabbitMQ Settings")
    credentials = pika.PlainCredentials(
        rabbitmq_settings.username, 
        rabbitmq_settings.password
    )
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host=rabbitmq_settings.host,
        port=int(rabbitmq_settings.port),
        virtual_host=rabbitmq_settings.virtual_host,
        credentials=credentials,
        heartbeat=600,                    # Keep connection alive
        blocked_connection_timeout=300    # Timeout for blocked connections
    ))
    return connection


def callback(ch, method, properties, body):
    """Process message with proper error handling and acknowledgment"""
    submission_data = None
    try:
        submission_data = json.loads(body)
        submission_id = submission_data.get("submission_id", "unknown")
        
        print(f"Processing submission: {submission_id}")
        
        # Process the submission
        process_image_submission(submission_data)
        
        # Only acknowledge AFTER successful processing
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f"Successfully processed: {submission_id}")
        
    except json.JSONDecodeError as e:
        # Invalid JSON - reject without requeue (dead letter)
        error_msg = f"Invalid JSON in message: {str(e)[:100]}"
        print(error_msg)
        ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
        try:
            frappe.log_error(message=error_msg, title="RabbitMQ JSON Error")
        except:
            pass  # Don't fail on logging errors
        
    except Exception as e:
        # Processing error - requeue for retry
        submission_id = submission_data.get("submission_id", "unknown") if submission_data else "unknown"
        error_msg = f"Error processing {submission_id}: {str(e)[:200]}"
        print(error_msg)
        
        try:
            frappe.log_error(message=error_msg, title=f"Process Error: {submission_id[:50]}")
        except:
            pass  # Don't fail on logging errors
        
        # Requeue the message for retry
        # Set requeue=False if you want to send to dead-letter queue instead
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)


def start_consuming():
    """Start consuming with proper configuration"""
    connection = None
    try:
        connection = connect_to_rabbitmq()
        channel = connection.channel()
        
        rabbitmq_settings = frappe.get_single("RabbitMQ Settings")
        queue_name = rabbitmq_settings.submission_queue
        
        # Use durable=False to match existing queue
        # If queue doesn't exist, this will create it with durable=False
        channel.queue_declare(queue=queue_name, durable=False)
        
        # Get queue stats using passive=True
        queue_info = channel.queue_declare(queue=queue_name, passive=True)
        message_count = queue_info.method.message_count
        print(f"Found {message_count} messages in queue '{queue_name}'")
        
        # CRITICAL: Set prefetch to 1 - process one message at a time
        channel.basic_qos(prefetch_count=1)
        
        # CRITICAL: auto_ack=False - manual acknowledgment after processing
        channel.basic_consume(
            queue=queue_name, 
            on_message_callback=callback, 
            auto_ack=False
        )
        
        print("RabbitMQ consumer started. Waiting for messages...")
        channel.start_consuming()
        
    except KeyboardInterrupt:
        print("RabbitMQ consumer stopped by user.")
    except Exception as e:
        error_msg = f"Error in RabbitMQ consumer: {str(e)[:200]}"
        print(error_msg)
        try:
            frappe.log_error(message=error_msg, title="RabbitMQ Consumer Error")
        except:
            pass  # Don't fail on logging errors
    finally:
        if connection and not connection.is_closed:
            connection.close()
            print("Connection closed.")


def get_queue_status():
    """Utility function to check queue status"""
    connection = None
    try:
        connection = connect_to_rabbitmq()
        channel = connection.channel()
        
        rabbitmq_settings = frappe.get_single("RabbitMQ Settings")
        queue_name = rabbitmq_settings.submission_queue
        
        queue_info = channel.queue_declare(queue=queue_name, passive=True)
        
        return {
            "queue": queue_name,
            "message_count": queue_info.method.message_count,
            "consumer_count": queue_info.method.consumer_count
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        if connection and not connection.is_closed:
            connection.close()
