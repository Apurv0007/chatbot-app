1 app.py — Main Flask Server
Runs your backend API
Connects the HTML page to your chatbot logic
Render.com uses this to start your application.

2.chatbot.py — Your Chatbot Brain
Contains generate_response()
All chatbot logic goes here
(You can expand it later).

 3.templates/index.html — Frontend Website
User chat interface
Calls the backend (/chat) to get responses
Must be inside a templates/ folder (Flask requirement).

4.requirements.txt — Dependencies
Render needs this to install Python libraries.
