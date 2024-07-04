import config as cfg
from google.cloud import aiplatform
from langchain_google_vertexai import VertexAIEmbeddings, VectorSearchVectorStore


aiplatform.init(
    project=cfg.PROJECT_ID, location=cfg.REGION, staging_bucket=cfg.BUCKET_URI
)

embedding_model = VertexAIEmbeddings(model_name="textembedding-gecko@003")


my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    location=cfg.REGION,
    display_name=cfg.DISPLAY_NAME,
    dimensions=cfg.DIMENSIONS,
    approximate_neighbors_count=cfg.ANN_COUNT,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
    index_update_method="BATCH_UPDATE",  # allowed values BATCH_UPDATE , STREAM_UPDATE
)

# Create an endpoint
my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name=f"{cfg.DISPLAY_NAME}-endpoint", public_endpoint_enabled=True
)

# NOTE : This operation can take upto 20 minutes
my_index_endpoint = my_index_endpoint.deploy_index(
    index=my_index, deployed_index_id=cfg.DEPLOYED_INDEX_ID
)

print(my_index_endpoint.deployed_indexes)


# NOTE : This operation can take more than 20 mins
vector_store = VectorSearchVectorStore.from_components(
    project_id=cfg.PROJECT_ID,
    region=cfg.REGION,
    gcs_bucket_name=cfg.BUCKET,
    index_id=my_index.name,
    endpoint_id=my_index_endpoint.name,
    embedding=embedding_model,
)


# Initialize the vectore_store as retriever
retriever = vector_store.as_retriever()


def embed(texts, metadatas):
    vector_store.add_texts(texts=texts, metadatas=metadatas, is_complete_overwrite=True)


def search(question):
    # perform simple similarity search on retriever
    result = retriever.invoke(question)
    return result
