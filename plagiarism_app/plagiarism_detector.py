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

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize image processing
image_directory = frappe.get_site_path('private', 'files', 'submitted_images')

if not os.path.exists(image_directory):
    os.makedirs(image_directory)

# Initialize ResNet model
resnet = models.resnet50(pretrained=True)
resnet.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def connect_to_feedback_queue() -> pika.BlockingConnection:
    """
    Establish connection to RabbitMQ feedback queue
    """
    try:
        rabbitmq_settings = frappe.get_single("RabbitMQ Settings")
        credentials = pika.PlainCredentials(
            rabbitmq_settings.username, 
            rabbitmq_settings.password
        )
        connection = pika.BlockingConnection(pika.ConnectionParameters(
            host=rabbitmq_settings.host,
            port=rabbitmq_settings.port,
            virtual_host=rabbitmq_settings.virtual_host,
            credentials=credentials
        ))
        return connection
    except Exception as e:
        logger.error(f"RabbitMQ connection error: {str(e)}")
        raise

def get_similar_sources(cluster_id: Optional[str]) -> List[Dict]:
    """
    Retrieve similar sources based on cluster_id
    Args:
        cluster_id: The ID of the cluster to fetch similar sources from
    Returns:
        List of dictionaries containing similar image metadata
    """
    if not cluster_id:
        return []
        
    try:
        similar_images = frappe.get_all(
            'Cluster Images',
            filters={'cluster_id': cluster_id},
            fields=['image_id', 'similarity_score', 'role_in_cluster']
        )
        
        sources = []
        for img in similar_images:
            try:
                image_meta = frappe.get_doc('Image Metadata', img.image_id)
                sources.append({
                    'submission_id': image_meta.submission_id,
                    'student_id': image_meta.student_id,
                    'assignment_id': image_meta.assignment_id,
                    'img_url': image_meta.original_url,
                    'similarity_score': img.similarity_score,
                    'role': img.role_in_cluster
                })
            except Exception as e:
                logger.error(f"Error fetching similar source {img.image_id}: {str(e)}")
        
        return sources
    except Exception as e:
        logger.error(f"Error getting similar sources for cluster {cluster_id}: {str(e)}")
        return []

def download_image(img_url: str, submission_id: str) -> str:
    """
    Download image from URL and save to local storage
    """
    try:
        response = requests.get(img_url)
        response.raise_for_status()
        
        image_path = os.path.join(image_directory, f"{submission_id}.jpg")
        with open(image_path, 'wb') as f:
            f.write(response.content)
            
        return image_path
    except Exception as e:
        logger.error(f"Error downloading image for {submission_id}: {str(e)}")
        raise

def attach_image_to_doc(image_path: str, submission_id: str) -> str:
    """
    Attach downloaded image to Frappe document
    """
    try:
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
    except Exception as e:
        logger.error(f"Error attaching image for {submission_id}: {str(e)}")
        raise

def extract_feature_vector(image_path: str) -> np.ndarray:
    """
    Extract feature vector from image using ResNet
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            features = resnet(image_tensor)
        return features.numpy().flatten()
    except Exception as e:
        logger.error(f"Error extracting feature vector: {str(e)}")
        raise

def check_for_plagiarism(image_id: str, feature_vector: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Check for similar images using FAISS
    """
    try:
        image_docs = frappe.get_all('Image Metadata', 
                                   fields=['feature_vector', 'name'])
        vectors = [json.loads(doc.feature_vector) for doc in image_docs 
                  if doc.feature_vector and doc.name != image_id]
        
        if not vectors:
            return None, None
            
        vectors = np.array(vectors).astype('float32')

        if vectors.shape[0] > 0:
            index = faiss.IndexFlatL2(len(feature_vector))
            index.add(vectors)

            distances, indices = index.search(
                np.array([feature_vector]).astype('float32'), 
                k=min(5, vectors.shape[0])
            )
            return distances[0], indices[0]
        return None, None
    except Exception as e:
        logger.error(f"Error checking plagiarism for {image_id}: {str(e)}")
        return None, None

def send_plagiarism_feedback(submission_id: str, 
                           student_id: str,
                           assignment_id: str,
                           img_url: str,
                           plagiarism_score: float,
                           similar_sources: List[Dict]) -> None:
    """
    Send plagiarism check results to RAG service
    """
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
            "plagiarism_score": plagiarism_score,
            "similar_sources": similar_sources
        }

        channel.basic_publish(
            exchange='',
            routing_key=rabbitmq_settings.feedback_queue,
            body=json.dumps(feedback_message),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        logger.info(f"Sent feedback for submission: {submission_id}")
        
    except Exception as e:
        logger.error(f"Error sending feedback for {submission_id}: {str(e)}")
    finally:
        if connection and not connection.is_closed:
            connection.close()

def process_image_submission(submission_data: Dict) -> None:
    """
    Process new image submission and check for plagiarism
    Args:
        submission_data: Dictionary containing submission details
        Expected format:
        {
            "submission_id": "IMSUB-xxx",
            "assign_id": "Make your cartoon-xxx",
            "student_id": "STxxx",
            "img_url": "https://..."
        }
    """
    try:
        submission_id = submission_data.get("submission_id")
        logger.info(f"Processing submission: {submission_id}")
        
        # Extract fields with direct mapping
        img_url = submission_data.get("img_url")
        student_id = submission_data.get("student_id")
        assign_id = submission_data.get("assign_id")

        # Validate required fields
        missing_fields = []
        if not submission_id:
            missing_fields.append("submission_id")
        if not img_url:
            missing_fields.append("img_url")
        if not student_id:
            missing_fields.append("student_id")
        if not assign_id:
            missing_fields.append("assign_id")

        if missing_fields:
            error_msg = f"Missing required fields: {', '.join(missing_fields)}"
            logger.error(error_msg)
            return

        # Process image
        image_path = download_image(img_url, submission_id)
        feature_vector = extract_feature_vector(image_path)
        file_url = attach_image_to_doc(image_path, submission_id)

        # Create metadata document
        image_doc = frappe.get_doc({
            "doctype": "Image Metadata",
            "submission_id": submission_id,
            "image_file": file_url,
            "original_url": img_url,
            "upload_date": frappe.utils.now_datetime(),
            "student_id": student_id,
            "assignment_id": assign_id,
            "feature_vector": json.dumps(feature_vector.tolist())
        })

        image_doc.insert(ignore_permissions=True)
        frappe.db.commit()
        logger.info(f"Created metadata for submission: {submission_id}")

        # Check for plagiarism
        distances, indices = check_for_plagiarism(image_doc.name, feature_vector)

        threshold = 0.95
        plagiarism_detected = False
        
        if distances is not None and len(distances) > 0:
            for i, distance in enumerate(distances):
                if distance < threshold:
                    plagiarism_detected = True
                    similar_image_doc = frappe.get_doc("Image Metadata", indices[i])
                    
                    # Create plagiarism flag
                    flag_doc = frappe.get_doc({
                        "doctype": "Plagiarism Flag",
                        "image_id": image_doc.name,
                        "cluster_id": similar_image_doc.cluster_id,
                        "flag_date": frappe.utils.now_datetime(),
                        "review_status": "Pending"
                    })
                    flag_doc.insert(ignore_permissions=True)
                    frappe.db.commit()
                    logger.info(f"Flagged potential plagiarism for: {submission_id}")

                    similar_sources = get_similar_sources(similar_image_doc.cluster_id)
                    send_plagiarism_feedback(
                        submission_id=submission_id,
                        student_id=student_id,
                        assignment_id=assign_id,
                        img_url=img_url,
                        plagiarism_score=float(distance),
                        similar_sources=similar_sources
                    )
                    break

        if not plagiarism_detected:
            send_plagiarism_feedback(
                submission_id=submission_id,
                student_id=student_id,
                assignment_id=assign_id,
                img_url=img_url,
                plagiarism_score=0.0,
                similar_sources=[]
            )

    except Exception as e:
        logger.error(f"Error processing submission {submission_id}: {str(e)}")
        frappe.db.rollback()
        
        if all([submission_id, student_id, assign_id, img_url]):
            send_plagiarism_feedback(
                submission_id=submission_id,
                student_id=student_id,
                assignment_id=assign_id,
                img_url=img_url,
                plagiarism_score=0.0,
                similar_sources=[]
            )
