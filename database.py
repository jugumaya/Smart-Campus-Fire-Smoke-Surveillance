import os
from typing import Optional

try:
    from motor.motor_asyncio import AsyncIOMotorClient
except ImportError as exc:
    raise ImportError("`motor` library is required. Install via: pip install motor") from exc

# Configure the local MongoDB metadata ingestion pipeline endpoint via environment variables
MONGO_DETAILS = os.getenv("MONGO_DETAILS", "mongodb://localhost:27017")

# Initialize the thread-safe asynchronous MongoDB client connection
client = AsyncIOMotorClient(MONGO_DETAILS)

# Target database instance and operational collection log spaces
database = client.smart_surveillance
logs_collection = database.get_collection("inference_logs")

async def save_inference_log(log_data: dict) -> Optional[str]:
    """
    Asynchronously persists real-time AI model inference logs into MongoDB.
    
    Args:
        log_data (dict): The inference payload containing bounding boxes, confidence, and asset links.
        
    Returns:
    Optional[str]: The uniquely generated MongoDB hex string ObjectId if successful, else None.
    """
    try:
        result = await logs_collection.insert_one(log_data)
        return str(result.inserted_id) if result else None
    except Exception as e:
        print(f"❌ Database Ingestion Failure: {e}")
        return None
    
    