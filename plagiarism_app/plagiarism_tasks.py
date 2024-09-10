from .plagiarism_detector import process_image_submission, get_faiss_index_path, load_faiss_index, save_faiss_index
from .celery_app import app

# Celery task for asynchronous processing of image submission
@app.task
def process_image_async(submission_data):
    # Get FAISS index path and ensure the directory exists
    faiss_index_path = get_faiss_index_path()

    # Load the FAISS index from the saved path
    faiss_index = load_faiss_index(faiss_index_path)

    # Process the image submission
    process_image_submission(submission_data, faiss_index)

    # Save the updated FAISS index after processing
    save_faiss_index(faiss_index, faiss_index_path)