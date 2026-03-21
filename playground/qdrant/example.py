import os
import dotenv
from qdrant_client import QdrantClient

dotenv.load_dotenv()

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_CLUSTER_END_POINT"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

print(qdrant_client.get_collections())
