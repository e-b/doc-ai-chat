PROJECT_ID = "genai-lab-390210"
REGION = "europe-west3"

# staging bucket
BUCKET = "impp_staging"
BUCKET_URI = f"gs://{BUCKET}"

# model
TEXT_MODEL = "gemini-1.5-pro"
EMBEDDING_MODEL = "textembedding-gecko@003"
# The number of dimensions for the textembedding-gecko@003 is 768
# If other embedder is used, the dimensions would probably need to change.
DIMENSIONS = 768

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Index Constants
DISPLAY_NAME = "impp_medical_qna"
DEPLOYED_INDEX_ID = "impp_medical_qna"

ANN_COUNT = 100
