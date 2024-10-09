import requests
import frappe
import cv2
import json
import numpy as np
import faiss
import os
import torch
import pika
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

image_directory = frappe.get_site_path('private', 'files', 'submitted_images')

if not os.path.exists(image_directory):
    os.makedirs(image_directory)

resnet = models.resnet50(pretrained=True)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def connect_to_feedback_queue():
    rabbitmq_settings = frappe.get_single("RabbitMQ Settings")
    credentials = pika.PlainCredentials(rabbitmq_settings.username, rabbitmq_settings.password)
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host=rabbitmq_settings.host,
        port=rabbitmq_settings.port,
        virtual_host=rabbitmq_settings.virtual_host,
        credentials=credentials))
    return connection

def send_plagiarism_feedback(image_id, plagiarism_flag, similarity_score, cluster_id):
    connection = connect_to_feedback_queue()
    channel = connection.channel()
    rabbitmq_settings = frappe.get_single("RabbitMQ Settings")
    channel.queue_declare(queue=rabbitmq_settings.feedback_queue, durable=True)

    if similarity_score is not None:
        similarity_score = float(similarity_score)

    feedback_message = {
        "image_id": image_id,
        "plagiarism_flag": plagiarism_flag,
        "similarity_score": similarity_score,
        "cluster_id": cluster_id
    }

    channel.basic_publish(
        exchange='',
        routing_key=rabbitmq_settings.feedback_queue,
        body=json.dumps(feedback_message),
        properties=pika.BasicProperties(
            delivery_mode=2
        )
    )
    
    connection.close()

def download_image(img_url, submission_id):
    response = requests.get(img_url)
    if response.status_code == 200:
        image_path = os.path.join(image_directory, f"{submission_id}.jpg")
        with open(image_path, 'wb') as f:
            f.write(response.content)
        return image_path
    else:
        frappe.throw(f"Failed to download image from {img_url}")

def attach_image_to_doc(image_path, submission_id):
    with open(image_path, 'rb') as filedata:
        file_doc = frappe.get_doc({
            "doctype": "File",
            "file_name": f"{submission_id}.jpg",
            "attached_to_doctype": "Image Metadata",
            "attached_to_name": submission_id,
            "is_private": 1,
            "content": filedata.read(),
        })
        file_doc.save()
        frappe.db.commit()
        return file_doc.file_url

def extract_feature_vector(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = resnet(image_tensor)
    return features.numpy().flatten()

def check_for_plagiarism(image_id, feature_vector):
    image_docs = frappe.get_all('Image Metadata', fields=['feature_vector', 'name'])
    vectors = [json.loads(doc.feature_vector) for doc in image_docs if doc.feature_vector]
    vectors = np.array(vectors).astype('float32')

    if vectors.shape[0] > 0:
        index = faiss.IndexFlatL2(len(feature_vector))
        index.add(vectors)

        distances, indices = index.search(np.array([feature_vector]).astype('float32'), k=5)
        return distances[0], indices[0]
    return None, None

def process_image_submission(submission_data):
    submission_id = submission_data.get("submission_id")
    img_url = submission_data.get("img_url")
    student_id = submission_data.get("student_id")

    if not img_url:
        frappe.logger().warning(f"Missing img_url for submission ID: {submission_id}, skipping...")
        return

    image_path = download_image(img_url, submission_id)
    feature_vector = extract_feature_vector(image_path)
    file_url = attach_image_to_doc(image_path, submission_id)

    image_doc = frappe.get_doc({
        "doctype": "Image Metadata",
        "submission_id": submission_id,
        "image_file": file_url,
        "upload_date": frappe.utils.now_datetime(),
        "student_id": student_id,
        "feature_vector": json.dumps(feature_vector.tolist())
    })
    image_doc.insert()
    frappe.db.commit()

    image_id = image_doc.name

    distances, indices = check_for_plagiarism(image_id, feature_vector)

    frappe.logger().info(f"Plagiarism Check - Distances: {distances}, Indices: {indices}")

    threshold = 0.95
    if distances is not None and len(distances) > 0:
        plagiarism_detected = False
        for i, distance in enumerate(distances):
            frappe.logger().info(f"Checking image {indices[i]} with distance {distance}")
            if distance < threshold:
                plagiarism_detected = True
                similar_image_doc = frappe.get_doc("Image Metadata", indices[i])
                frappe.get_doc({
                    "doctype": "Plagiarism Flag",
                    "image_id": image_id,
                    "cluster_id": similar_image_doc.cluster_id,
                    "flag_date": frappe.utils.now_datetime(),
                    "review_status": "Pending"
                }).insert()

                send_plagiarism_feedback(
                    image_id=image_id,
                    plagiarism_flag="Pending",
                    similarity_score=distance,
                    cluster_id=similar_image_doc.cluster_id
                )

        if not plagiarism_detected:
            send_plagiarism_feedback(
                image_id=image_id,
                plagiarism_flag="No Plagiarism",
                similarity_score=None,
                cluster_id=None
            )

    else:
        frappe.logger().info(f"No similar images found for image {image_id}")

        send_plagiarism_feedback(
            image_id=image_id,
            plagiarism_flag="No Plagiarism",
            similarity_score=None,
            cluster_id=None
        )
