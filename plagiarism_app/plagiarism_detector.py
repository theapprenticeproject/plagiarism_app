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
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Directory for saving downloaded images
image_directory = frappe.get_site_path('private', 'files', 'submitted_images')
os.makedirs(image_directory, exist_ok=True)

# Load pretrained ResNet50
resnet = models.resnet50(pretrained=True)
resnet.eval()

# Transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def convert_to_json_serializable(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    else:
        return obj

def connect_to_feedback_queue() -> pika.BlockingConnection:
    rabbitmq_settings = frappe.get_single("RabbitMQ Settings")
    credentials = pika.PlainCredentials(rabbitmq_settings.username, rabbitmq_settings.password)
    return pika.BlockingConnection(pika.ConnectionParameters(
        host=rabbitmq_settings.host,
        port=rabbitmq_settings.port,
        virtual_host=rabbitmq_settings.virtual_host,
        credentials=credentials
    ))

def download_image(img_url: str, submission_id: str) -> str:
    try:
        image_path = os.path.join(image_directory, f"{submission_id}.jpg")
        if img_url.startswith('file:///'):
            local_path = img_url.replace('file:///', '')
            image = Image.open(local_path).convert('RGB')
            image.save(image_path, format='JPEG')
        else:
            response = requests.get(img_url)
            response.raise_for_status()
            with open(image_path, 'wb') as f:
                f.write(response.content)
        return image_path
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        raise

def attach_image_to_doc(image_path: str, submission_id: str) -> str:
    with open(image_path, 'rb') as filedata:
        file_doc = frappe.get_doc({
            "doctype": "File",
            "file_name": f"{submission_id}.jpg",
            "attached_to_doctype": "Image Metadata",
            "attached_to_name": submission_id,
            "is_private": 1,
            "content": filedata.read(),
        })
        file_doc.insert(ignore_permissions=True)
        frappe.db.commit()
        return file_doc.file_url

def extract_feature_vectors(image_path: str) -> List[np.ndarray]:
    base_image = Image.open(image_path).convert('RGB')
    base_image = base_image.resize((224, 224), Image.Resampling.LANCZOS)
    angles = [0, 90, 180, 270]
    vectors = []
    for angle in angles:
        rotated_image = base_image.rotate(angle)
        image_tensor = transform(rotated_image).unsqueeze(0)
        with torch.no_grad():
            features = resnet(image_tensor)
        vector = features.numpy().flatten()
        vector = vector / np.linalg.norm(vector)
        vectors.append(vector)
    return vectors

def check_for_plagiarism(image_id: str, submission_id: str, rotated_vectors: List[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[list], Optional[List[int]]]:
    try:
        image_docs = frappe.get_all('Image Metadata', fields=['feature_vector', 'name', 'submission_id'])
        stored_vectors, doc_names = [], []

        for doc in image_docs:
            # Avoid comparing with self (by both name and submission_id)
            if doc.feature_vector and doc.name != image_id and doc.submission_id != submission_id:
                vector = np.array(json.loads(doc.feature_vector)).astype('float32')
                stored_vectors.append(vector)
                doc_names.append(doc.name)

        if not stored_vectors:
            return None, None, None

        stored_vectors = np.array(stored_vectors).astype('float32')
        index = faiss.IndexFlatL2(stored_vectors.shape[1])
        index.add(stored_vectors)

        best_score = 0
        best_index = None
        best_distance = None
        exact_match_found = False

        for rotated_vec in rotated_vectors:
            rotated_vec = rotated_vec.astype('float32')
            k = min(5, len(stored_vectors))
            distances, indices = index.search(np.array([rotated_vec]), k=k)

            for dist, idx in zip(distances[0], indices[0]):
                stored_vec = stored_vectors[idx]
                cos_sim = np.dot(rotated_vec, stored_vec) / (np.linalg.norm(rotated_vec) * np.linalg.norm(stored_vec))
                if cos_sim >= 0.9999:
                    similarity = 1
                    exact_match_found = True
                else:
                    similarity = 0

                if similarity > best_score:
                    best_score = similarity
                    best_index = idx
                    best_distance = dist

        if best_index is not None:
            doc_indices = [doc_names[best_index]]
            similarity_scores = [1 if exact_match_found else 0]
            return [best_distance], doc_indices, similarity_scores
        else:
            return None, None, None

    except Exception as e:
        logger.error(f"Error checking plagiarism: {str(e)}")
        return None, None, None

def send_plagiarism_feedback(submission_id: str, student_id: str, assignment_id: str,
                              img_url: str, plagiarism_score: int,
                              similar_sources: List[Dict]) -> None:
    connection = None
    try:
        connection = connect_to_feedback_queue()
        channel = connection.channel()
        rabbitmq_settings = frappe.get_single("RabbitMQ Settings")
        channel.queue_declare(queue=rabbitmq_settings.feedback_queue, durable=True)
        feedback_message = {
            "submission_id": submission_id,
            "student_id": student_id,
            "assignment_id": assignment_id,
            "img_url": img_url,
            "plagiarism_score": int(plagiarism_score),
            "similar_sources": convert_to_json_serializable(similar_sources)
        }
        channel.basic_publish(
            exchange='',
            routing_key=rabbitmq_settings.feedback_queue,
            body=json.dumps(feedback_message),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        logger.info(f"Feedback sent for submission: {submission_id}")
    except Exception as e:
        logger.error(f"Error sending feedback: {str(e)}")
    finally:
        if connection and not connection.is_closed:
            connection.close()

def process_image_submission(submission_data: Dict) -> None:
    try:
        submission_id = submission_data.get("submission_id")
        img_url = submission_data.get("img_url")
        student_id = submission_data.get("student_id")
        assign_id = submission_data.get("assign_id")

        missing = [key for key in ['submission_id', 'img_url', 'student_id', 'assign_id'] if not submission_data.get(key)]
        if missing:
            logger.error(f"Missing fields: {', '.join(missing)}")
            return

        image_path = download_image(img_url, submission_id)
        rotated_vectors = extract_feature_vectors(image_path)

        # Do plagiarism check BEFORE inserting the new image
        distances, doc_indices, similarity_scores = check_for_plagiarism(
            image_id=submission_id, submission_id=submission_id, rotated_vectors=rotated_vectors
        )

        file_url = attach_image_to_doc(image_path, submission_id)

        # Save the image only after checking
        image_doc = frappe.get_doc({
            "doctype": "Image Metadata",
            "submission_id": submission_id,
            "image_file": file_url,
            "original_url": img_url,
            "upload_date": frappe.utils.now_datetime(),
            "student_id": student_id,
            "assignment_id": assign_id,
            "feature_vector": json.dumps(rotated_vectors[0].tolist())
        })
        image_doc.insert(ignore_permissions=True)
        frappe.db.commit()

        similar_sources = []
        plagiarism_score = max(similarity_scores) if similarity_scores else 0

        if similarity_scores:
            for i, similarity in enumerate(similarity_scores):
                similar_doc = frappe.get_doc("Image Metadata", doc_indices[i])
                flag = frappe.get_doc({
                    "doctype": "Plagiarism Flag",
                    "image_id": image_doc.name,
                    "cluster_id": getattr(similar_doc, 'cluster_id', None),
                    "similarity_score": int(similarity),
                    "custom_plagiarism_score": int(plagiarism_score),
                    "flag_date": frappe.utils.now_datetime(),
                    "review_status": "Pending"
                })
                flag.insert(ignore_permissions=True)
                frappe.db.commit()
                similar_sources.append({
                    'submission_id': similar_doc.submission_id,
                    'student_id': similar_doc.student_id,
                    'assignment_id': similar_doc.assignment_id,
                    'img_url': similar_doc.original_url,
                    'similarity_score': int(similarity),
                    'role': 'duplicate'
                })

        send_plagiarism_feedback(submission_id, student_id, assign_id, img_url, plagiarism_score, similar_sources)

    except Exception as e:
        logger.error(f"Error processing {submission_data.get('submission_id')}: {str(e)}")
        frappe.db.rollback()
        if all(k in submission_data for k in ['submission_id', 'student_id', 'assign_id', 'img_url']):
            send_plagiarism_feedback(
                submission_id=submission_data['submission_id'],
                student_id=submission_data['student_id'],
                assignment_id=submission_data['assign_id'],
                img_url=submission_data['img_url'],
                plagiarism_score=0,
                similar_sources=[]
            )
