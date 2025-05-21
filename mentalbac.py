import ollama
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
import uuid
import re
from datetime import datetime, timedelta

# Qdrant configuration
qdrant_config = {
    "url": "http://localhost:6333",
    "user_collection": "users",
    "chat_collection": "chat_history"
}

# Initialize Qdrant client and collections
def init_qdrant():
    try:
        client = QdrantClient(url=qdrant_config["url"])
        collections = client.get_collections().collections
        if qdrant_config["user_collection"] not in [c.name for c in collections]:
            client.create_collection(
                collection_name=qdrant_config["user_collection"],
                vectors_config=VectorParams(size=3, distance=Distance.COSINE)
            )
        if qdrant_config["chat_collection"] not in [c.name for c in collections]:
            client.create_collection(
                collection_name=qdrant_config["chat_collection"],
                vectors_config=VectorParams(size=3, distance=Distance.COSINE)
            )
        return client
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        return None

# Validate email and phone number
def validate_input(email, phone_number):
    email_pattern = r"[^@]+@[^@]+\.[^@]+"
    phone_pattern = r"^\+?\d{10,15}$"
    if not re.match(email_pattern, email):
        return False, "Invalid email format."
    if not re.match(phone_pattern, phone_number):
        return False, "Invalid phone number format (10-15 digits, optional +)."
    return True, ""

# Store chat history
def store_chat_history(client, user_id, user_input, bot_response):
    try:
        chat_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        dummy_vector = [0, 0, 0]
        payload = {
            "user_id": user_id,
            "user_input": user_input,
            "bot_response": bot_response,
            "timestamp": timestamp
        }
        client.upsert(
            collection_name=qdrant_config["chat_collection"],
            points=[
                PointStruct(
                    id=chat_id,
                    vector=dummy_vector,
                    payload=payload
                )
            ]
        )
    except Exception as e:
        print(f"Error storing chat history: {e}")

# Retrieve last conversation within one month
def get_last_conversation(client, user_id):
    try:
        one_month_ago = (datetime.utcnow() - timedelta(days=30)).isoformat()
        search_result = client.scroll(
            collection_name=qdrant_config["chat_collection"],
            scroll_filter={
                "must": [
                    {"key": "user_id", "match": {"value": user_id}},
                    {"key": "timestamp", "range": {"gte": one_month_ago}}
                ]
            },
            limit=1,
            with_payload=True,
            order_by={"key": "timestamp", "direction": "desc"}
        )[0]
        if search_result:
            return search_result[0].payload
        return None
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return None

# Generate dynamic follow-up question without fallback
def generate_follow_up_question(last_conversation):
    if not last_conversation:
        return None
    user_input = last_conversation["user_input"].lower()
    concern_map = {
        "exam": "Last time you mentioned feeling worried about your exams. How are you feeling about them now?",
        "lonely": "Last time you shared that you were feeling lonely. How are things going with your connections now?",
        "stress": "You mentioned feeling stressed last time. How are you managing that stress now?",
        "sad": "You talked about feeling sad before. How are you feeling today?",
        "anxious": "Last time you felt anxious. Are you still experiencing those feelings?",
        "depressed": "You mentioned feeling depressed previously. How are you doing now?"
    }
    for concern, question in concern_map.items():
        if concern in user_input:
            return question
    return None

# Check if user exists and store new user
def handle_user():
    client = init_qdrant()
    if not client:
        return None, None, "Qdrant connection failed. Please try again later."

    print("Please provide your phone number or email to continue.")
    identifier = input("Phone number or email: ").strip().lower()

    search_result = client.scroll(
        collection_name=qdrant_config["user_collection"],
        scroll_filter={
            "should": [
                {"key": "phone_number", "match": {"value": identifier}},
                {"key": "email", "match": {"value": identifier}}
            ]
        },
        limit=1
    )[0]

    if search_result:
        user = search_result[0].payload
        user_id = search_result[0].id
        name = user["name"]
        last_conversation = get_last_conversation(client, user_id)
        follow_up = generate_follow_up_question(last_conversation)
        welcome_message = f"Welcome back, {name}! I'm here to support you with any mental health concerns."
        if follow_up:
            welcome_message += f" {follow_up}"
        else:
            welcome_message += " What's on your mind?"
        return user_id, name, welcome_message
    else:
        print("It looks like you're new! Please provide some details.")
        name = input("Your name: ").strip()
        phone_number = input("Your phone number: ").strip()
        email = input("Your email: ").strip().lower()

        is_valid, error_message = validate_input(email, phone_number)
        if not is_valid:
            return None, None, error_message

        try:
            user_id = str(uuid.uuid4())
            dummy_vector = [0, 0, 0]
            payload = {
                "name": name,
                "phone_number": phone_number,
                "email": email
            }
            client.upsert(
                collection_name=qdrant_config["user_collection"],
                points=[
                    PointStruct(
                        id=user_id,
                        vector=dummy_vector,
                        payload=payload
                    )
                ]
            )
            welcome_message = f"Welcome, {name}! Thank you for sharing your details. I'm here to help with any mental health concerns. What's on your mind?"
            return user_id, name, welcome_message
        except Exception as e:
            return None, None, f"Error saving user data: {e}"

# Generate response from the model
def chat_with_model(user_input, user_name):
    try:
        response = ollama.chat(
            model="llama3.2-vision:11b",
            messages=[
                {
                    "role": "system",
                    "content": f"""
                    You are a compassionate psychiatrist specializing in adolescent mental health, designed exclusively to support students with emotional concerns like loneliness, guilt, or stress. Address the user as {user_name}. Respond with empathy, clarity, and professionalism, using the following structure:

                    1. **Greeting and Validation**: Begin with a warm, supportive greeting. Validate the student’s specific emotions in a non-judgmental tone.
                    2. **Psychological Insight**: Explain the emotional or cognitive basis of their feelings using simple, neuroscience-based terms.
                    3. **Actionable Advice**: Provide 2-3 specific, practical strategies to address their feelings and foster resilience or connection.
                    4. **Growth and Encouragement**: Emphasize that their feelings are normal, their worth is not defined by their situation, and small steps lead to progress. End with a kind, empowering message.
                    5. **Counselor Recommendation**: If feelings seem persistent or severe, gently suggest talking to a school counselor or trusted adult.

                    **Rules**:
                    - **Tone**: Use clear, concise language with a professional yet approachable tone, like a trusted adult. Avoid emojis, slang, or clinical jargon.
                    - **Audience**: Respond only to students with mental health concerns. For non-students, say: “Sorry, I’m designed to help students only.”
                    - **Scope**: Address only student mental health topics. For off-topic queries, say: “Sorry, I can only help with student mental health concerns.”
                    - **Safety**: Never offer medical advice (e.g., medication).
                    """
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        )
        if any(keyword in user_input.lower() for keyword in ["parent", "employee", "teacher"]):
            return "Sorry, I was made especially to help students only. I can't answer that."
        if not any(keyword in user_input.lower() for keyword in ["feel", "stress", "lonely", "sad", "anxious", "depressed", "mental"]):
            return "Sorry, I can only talk about student mental health. Please ask something related to that."
        
        return response['message']['content']
    except Exception as e:
        return f"Error: {e}"

# Main chat loop
def main():
    print("Welcome my friend! I'm here to support students with mental health concerns.")
    
    client = init_qdrant()
    if not client:
        print("Qdrant connection failed. Please try again later.")
        return
    
    user_id, user_name, welcome_message = handle_user()
    if not user_id:
        print(welcome_message)
        return
    
    print(welcome_message)
    
    while True:
        user_input = input("> ").strip()
        if user_input.lower() == "quit":
            print(f"Goodbye, {user_name}! Take care.")
            break
        response = chat_with_model(user_input, user_name)
        print(f"Psychiatrist: {response}")
        store_chat_history(client, user_id, user_input, response)

if __name__ == "__main__":
    main()