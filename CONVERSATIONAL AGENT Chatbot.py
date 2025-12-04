#!/usr/bin/env python
# coding: utf-8

# # Installing & Importing libraries

# In[1]:


# If needed, run this once in a Jupyter cell
#!pip install pandas scikit-learn nltk seaborn matplotlib

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

# Correct seaborn styling (NOT plt.style.use("seaborn"))
sns.set_theme(style="whitegrid")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from collections import deque, Counter

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

EN_STOPWORDS = set(stopwords.words('english'))


import warnings
warnings.filterwarnings("ignore")


# # Loading dataset (multilingual-safe)

# In[2]:


def load_dataset(path="dataset.csv"):
    # encoding='utf-8-sig' helps if file has BOM or non-ASCII text
    df = pd.read_csv(path, encoding='utf-8-sig')

    expected_cols = {
        "Rating", "Review", "Product Name",
        "Product Category", "Emotion",
        "Data Source", "Sentiment"
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # Remove rows where review or labels are missing
    df = df.dropna(subset=["Review", "Emotion", "Sentiment"])

    return df

df = load_dataset("dataset.csv")
df.head()


# ### Dataset overview

# In[3]:


df.info()


# ### Checking missing values

# In[4]:


df.isnull().sum()


# ### Basic text statistics

# In[5]:


df["review_length"] = df["Review"].astype(str).apply(len)
df["word_count"] = df["Review"].astype(str).apply(lambda x: len(x.split()))

df[["review_length", "word_count"]].describe()


# # Visualization

# ### Rating distribution

# In[6]:


plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Rating", palette="viridis")
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()


# ### Emotion distribution

# In[7]:


plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Emotion", palette="coolwarm")
plt.title("Emotion Label Distribution")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


# ### Sentiment distribution

# In[8]:


plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Sentiment", palette="Spectral")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()


# ### Review length distribution

# In[9]:


plt.figure(figsize=(8,5))
sns.histplot(df["word_count"], bins=50, kde=True, color="blue")
plt.title("Word Count Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()


# ### Emotion vs Review Length

# In[10]:


plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="Emotion", y="word_count", palette="Set2")
plt.title("Word Count by Emotion Type")
plt.xlabel("Emotion")
plt.ylabel("Word Count")
plt.xticks(rotation=45)
plt.show()


# ### Top words per emotion

# In[11]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words="english")
counts = vectorizer.fit_transform(df["Review"].astype(str))

words = vectorizer.get_feature_names_out()

emotion_groups = df.groupby("Emotion")

for emotion, group in emotion_groups:
    text = group["Review"].astype(str)
    vec = vectorizer.fit_transform(text)
    sum_words = vec.sum(axis=0)

    word_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)

    print(f"\nTop words for Emotion: {emotion}")
    print(word_freq[:10])


# In[ ]:





# # Multilingual-friendly text cleaning

# In[12]:


def clean_text_multilingual(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Optional: remove English stopwords where applicable
    tokens = text.split()
    cleaned_tokens = []
    for t in tokens:
        # Heuristic: if token is pure ASCII letters, treat it as English
        if re.fullmatch(r"[a-zA-Z]+", t) and t in EN_STOPWORDS:
            continue
        cleaned_tokens.append(t)

    return " ".join(cleaned_tokens)

df["Review_clean"] = df["Review"].apply(clean_text_multilingual)
df[["Review", "Review_clean"]].head()


# # Emotion & Sentiment models (TF-IDF + Linear SVM)

# In[13]:


class EmotionSentimentModels:
    def __init__(self):
        self.emotion_clf = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=30000,
                ngram_range=(1, 2)
            )),
            ("clf", LinearSVC())
        ])

        self.sentiment_clf = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=30000,
                ngram_range=(1, 2)
            )),
            ("clf", LinearSVC())
        ])

        self.trained = False

    def train(self, df: pd.DataFrame, test_size=0.2, random_state=42):
        X = df["Review_clean"].values
        y_emotion = df["Emotion"].values
        y_sentiment = df["Sentiment"].values

        # Emotion
        X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
            X, y_emotion, test_size=test_size,
            random_state=random_state, stratify=y_emotion
        )

        print("Training emotion classifier...")
        self.emotion_clf.fit(X_train_e, y_train_e)
        y_pred_e = self.emotion_clf.predict(X_test_e)
        print("\n=== Emotion Classification Report ===")
        print(classification_report(y_test_e, y_pred_e))
        print("Emotion Accuracy:", accuracy_score(y_test_e, y_pred_e))

        # Sentiment
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X, y_sentiment, test_size=test_size,
            random_state=random_state, stratify=y_sentiment
        )

        print("\nTraining sentiment classifier...")
        self.sentiment_clf.fit(X_train_s, y_train_s)
        y_pred_s = self.sentiment_clf.predict(X_test_s)
        print("\n=== Sentiment Classification Report ===")
        print(classification_report(y_test_s, y_pred_s))
        print("Sentiment Accuracy:", accuracy_score(y_test_s, y_pred_s))

        self.trained = True

    def predict(self, text: str):
        if not self.trained:
            raise RuntimeError("Models are not trained yet.")

        clean = clean_text_multilingual(text)
        emotion = self.emotion_clf.predict([clean])[0]
        sentiment = self.sentiment_clf.predict([clean])[0]
        return emotion, sentiment


# # Affective memory (neuro-inspired internal state)

# In[14]:


class AffectiveMemory:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.recent_emotions = deque(maxlen=window_size)
        self.recent_sentiments = deque(maxlen=window_size)

    def update(self, emotion: str, sentiment: str):
        self.recent_emotions.append(emotion)
        self.recent_sentiments.append(sentiment)

    def dominant_emotion(self):
        if not self.recent_emotions:
            return None
        return Counter(self.recent_emotions).most_common(1)[0][0]

    def dominant_sentiment(self):
        if not self.recent_sentiments:
            return None
        return Counter(self.recent_sentiments).most_common(1)[0][0]

    def emotional_context(self):
        dom_e = self.dominant_emotion()
        if dom_e is None:
            return "neutral context"

        if dom_e in ["Anger", "Fear", "Sadness"]:
            return "user is in a generally negative emotional state"
        elif dom_e in ["Happiness", "Love", "Happy"]:  # include your label names
            return "user is in a generally positive and warm state"
        else:
            return f"user has mostly shown {dom_e} recently"


# # Neuro-inspired conversational agent

# In[15]:


class NeuroInspiredAgent:
    def __init__(self, models: EmotionSentimentModels, memory: AffectiveMemory):
        self.models = models
        self.memory = memory

    def appraise_state(self, emotion: str, sentiment: str) -> str:
        sent = sentiment.lower()
        if "neg" in sent:
            if emotion in ["Anger", "Fear"]:
                return "highly distressed or frustrated"
            elif emotion in ["Sad", "Sadness"]:
                return "sad or emotionally low"
            else:
                return "unhappy with the experience"
        elif "pos" in sent:
            if emotion in ["Happy", "Happiness", "Love"]:
                return "very satisfied and emotionally positive"
            else:
                return "moderately satisfied"
        else:
            return "neutral or mixed feelings"

    def generate_empathetic_reply(self, user_text: str) -> str:
        # 1. Predict emotion & sentiment
        emotion, sentiment = self.models.predict(user_text)
        self.memory.update(emotion, sentiment)

        # 2. Cognitive appraisal & affective context
        appraisal = self.appraise_state(emotion, sentiment)
        context_summary = self.memory.emotional_context()

        # 3. Empathetic prefix based on emotional state
        if emotion in ["Sad", "Sadness", "Fear", "Anger"] and "neg" in sentiment.lower():
            prefix = (
                "I'm really sorry that you're feeling this way. "
                "It sounds like this experience may have been difficult or upsetting for you."
            )
        elif emotion in ["Happy", "Happiness", "Love"] and "pos" in sentiment.lower():
            prefix = (
                "I'm glad to hear that you're feeling positive about this. "
                "It's great when experiences or products genuinely make you feel good."
            )
        else:
            prefix = (
                "Thank you for sharing your experience. "
                "Your feedback gives me a clearer understanding of how you feel."
            )

        # 4. Integrate affective memory & appraisal
        context_part = f"\n\nFrom what you've shared so far, it seems that {context_summary}."
        appraisal_part = f"\nOverall, you appear to be {appraisal}."

        # 5. Gentle closing prompt
        closing = (
            "\n\nIf you'd like, you can tell me more about what mattered most "
            "to you in this situation, and I’ll respond accordingly."
        )

        # 6. Final response
        return (
            f"[Detected Emotion: {emotion} | Sentiment: {sentiment}]\n\n"
            f"{prefix}{appraisal_part}{context_part}{closing}"
        )

    def reply_once(self, user_text: str) -> str:
        return self.generate_empathetic_reply(user_text)


# # Train models & creating agent

# In[16]:


# Assume df already loaded and df["Review_clean"] is prepared

models = EmotionSentimentModels()
models.train(df)   # This will print classification reports

memory = AffectiveMemory(window_size=10)
agent = NeuroInspiredAgent(models, memory)

print("Agent is ready to chat.")


# ## Quick single test

# In[17]:


sample_text = "I really love this phone. The performance is smooth and the battery life is amazing!"
response = agent.reply_once(sample_text)
print(response)


# # Interactive Chatbot loop

# In[18]:


print("Emotional Chat Agent (type 'exit' or 'quit' to stop)\n")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("\nAgent: Thank you for talking with me. Take care. 💬")
        break

    reply = agent.reply_once(user_input)
    print("\nAgent:")
    print(reply)
    print()


# In[ ]:





# # Full Conversational LLM using Gemini Model

# In[29]:


#!pip install -U google-generativeai


# In[20]:


import os
os.environ["GEMINI_API_KEY"] = "AIzaSyA28vCgo3ogAyPnb9FP6bWDhW1Zd9M5o1g" 


# In[ ]:





# In[23]:


import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# See what models your account can use
for m in genai.list_models():
    print(m.name)


# In[24]:


import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")  # or "gemini-2.5-pro"
response = model.generate_content("Say hello in one short sentence.")
print(response.text)


# ## NeuroInspiredAgent with Gemini

# In[25]:


class NeuroInspiredAgent:
    def __init__(self, models: EmotionSentimentModels, memory: AffectiveMemory):
        self.models = models
        self.memory = memory
        # Use a current Gemini model
        self.llm = genai.GenerativeModel("gemini-2.5-flash")  # or "gemini-2.5-pro"

    def appraise_state(self, emotion: str, sentiment: str) -> str:
        sent = sentiment.lower()
        if "neg" in sent:
            if emotion in ["Anger", "Fear"]:
                return "highly distressed or frustrated"
            elif emotion in ["Sad", "Sadness"]:
                return "sad or emotionally low"
            else:
                return "unhappy with the experience"
        elif "pos" in sent:
            if emotion in ["Happy", "Happiness", "Love"]:
                return "very satisfied and emotionally positive"
            else:
                return "moderately satisfied"
        else:
            return "neutral or mixed feelings"

    def generate_empathetic_reply(self, user_text: str) -> str:
        # 1. Predict emotion & sentiment
        emotion, sentiment = self.models.predict(user_text)
        self.memory.update(emotion, sentiment)

        # 2. Cognitive appraisal & affective context
        appraisal = self.appraise_state(emotion, sentiment)
        context_summary = self.memory.emotional_context()

        # 3. Build prompt for Gemini
        system_prompt = (
            "You are an emotionally intelligent, empathetic conversational agent. "
            "Respond in clear, kind English. Be supportive, specific, and non-judgmental. "
            "Acknowledge the user's feelings explicitly and stay concise (5–8 sentences). "
            "Do not invent product facts; focus on emotions and experience."
        )

        user_conditioning = f"""
User message:
\"\"\"{user_text}\"\"\"

Detected emotional state:
- Emotion label: {emotion}
- Sentiment label: {sentiment}
- Cognitive appraisal: {appraisal}
- Affective context (recent interactions): {context_summary}

Using this information:
1. Validate or acknowledge how the user feels.
2. Reflect their emotional state in natural language.
3. Offer a calm, empathetic, and context-aware response.
4. If they are distressed or unhappy, be especially gentle.
5. If they are happy or satisfied, recognise and reinforce that positive feeling.

Reply in English only.
"""

        llm_response = self.llm.generate_content(
            [system_prompt, user_conditioning]
        )
        text_reply = llm_response.text.strip()

        final_reply = (
            f"[Detected Emotion: {emotion} | Sentiment: {sentiment} | Appraisal: {appraisal}]\n\n"
            f"{text_reply}"
        )

        return final_reply

    def reply_once(self, user_text: str) -> str:
        return self.generate_empathetic_reply(user_text)


# # Train classical models & create agent

# In[26]:


models = EmotionSentimentModels()
models.train(df)   # trains TF-IDF + LinearSVC for Emotion & Sentiment

memory = AffectiveMemory(window_size=10)
agent = NeuroInspiredAgent(models, memory)

print("Gemini-powered neuro-inspired agent is ready.")


# ## Quick test

# In[27]:


sample_text = "I am very disappointed with this product. It stopped working after two days."
reply = agent.reply_once(sample_text)
print(reply)


# # Interactive chat loop

# In[28]:


print("Gemini Emotional Chat Agent (type 'exit' or 'quit' to stop)\n")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("\nAgent: Thank you for talking with me. Take care. 💬")
        break

    response = agent.reply_once(user_input)
    print("\nAgent:")
    print(response)
    print()


# In[ ]:




