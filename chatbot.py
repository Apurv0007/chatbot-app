def generate_response(user_input):
    user_input = user_input.lower()

    if "hello" in user_input or "hi" in user_input:
        return "Hello! How can I help you today?"
    elif "your name" in user_input:
        return "I'm your conversational chatbot!"
    elif "help" in user_input:
        return "Sure, tell me what you need help with."
    else:
        return "I'm not sure about that, but I'm learning!"
