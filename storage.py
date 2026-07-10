import io
import time
from typing import Optional
from dotenv import load_dotenv
from supabase import create_client, Client

# Ingest underlying environment configurations securely
load_dotenv()

# Cloud storage access configuration credentials
SUPABASE_URL = "https://npnekpfqnjusyjiakjng.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5wbmVrcGZxbmp1c3lqaWFram5nIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODAzMTY1MzcsImV4cCI6MjA5NTg5MjUzN30.48WyZt5LC6YemZ2_jOM-riJIXq60hn52zQl-4H8B8UA"
BUCKET_NAME = "inference-images"

# Instantiate a single connection context to the Supabase client architecture
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

async def upload_image_to_cloud(file_bytes: bytes, filename: str) -> Optional[str]:
    """
    Uploads processed threat visualization frames directly into Supabase Storage.
    
    Args:
        file_bytes (bytes): Binary streams representing the annotated image.
        filename (str): Original tracking file name context.
        
    Returns:
        Optional[str]: Fully qualified asset distribution URL link, or None if connection drops.
    """
    try:
        # Avoid file collision inside cloud storage by prepending epoch timestamp
        unique_filename = f"{int(time.time())}_{filename}"
        file_buffer = io.BytesIO(file_bytes)
        raw_data = file_buffer.getvalue()

        # Fire asynchronous storage payload transmission request onto the target bucket
        supabase.storage.from_(BUCKET_NAME).upload(
            path=unique_filename,
            file=raw_data,
            file_options={"content-type": "image/jpeg"}
        )

        # Retrieve the static public asset link for the dashboard visualization interface
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(unique_filename)
        return public_url
    except Exception as e:
        print(f"❌ Supabase Asset Injection Failure: {str(e)}")
        return None
    
    