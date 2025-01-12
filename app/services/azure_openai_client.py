import os

import openai
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from app.core import ai_search_helper
from app.model import Prompts
import json
from random import choice
import asyncio
from typing import List, Dict
import nltk
from nltk.corpus import stopwords,words
from nltk.tokenize import word_tokenize
import re
import logging

logger = logging.getLogger(__name__)

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


# Download the NLTK words corpus
nltk.download('words')

# Set of valid English words
english_words = set(words.words())

def clean_text(text: str) -> str:
    """Remove stop words and unnecessary spaces using a regex-based tokenizer"""
    stop_words = set(stopwords.words('english'))
    # Use regex to extract words
    word_tokens = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)

def calculate_similarity_score(current_query: str, history_entry: Dict) -> float:
    """Calculate semantic similarity between current query and history entry"""
    query_tokens = set(clean_text(current_query).split())
    history_tokens = set(clean_text(history_entry['content']).split())
    
    if not query_tokens or not history_tokens:
        return 0.0
    
    intersection = query_tokens.intersection(history_tokens)
    return len(intersection) / (len(query_tokens) + len(history_tokens) - len(intersection))

class OpenAIClient:
    index_name = os.environ.get("AZURE_AI_SEARCH_INDEX")
    search_key = os.environ.get("AZURE_AI_SEARCH_KEY")
    search_endpoint = os.environ.get("AZURE_AI_SEARCH_ENDPOINT")
    chat_history = {}
    prompt = ""

    def __init__(self):
        openai.api_type = os.environ.get("OPENAI_API_TYPE")
        openai.api_base = os.environ.get("OPENAI_API_BASE")
        openai.api_version = os.environ.get("OPENAI_API_VERSION")
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    def get_search_connection(self, index=index_name):
        credential = AzureKeyCredential(self.search_key)
        client = SearchClient(
            endpoint=self.search_endpoint, index_name=index, credential=credential
        )
        return client

    def get_completion(self, prompt, model="gpt-4o"):
        messages = [{"role": "user", "content": prompt}]
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,  # this is the degree of randomness of the model's output
        )
        return response.choices[0].message.content

    def get_completion_from_messages(self, messages, model="gpt-4o", temperature=0):
        response = openai.AsyncAzureOpenAI.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,  # this is the degree of randomness of the model's output
        )
        return response.choices[0].message.content

    def get_context_from_azure_search(
        self, search_text, index=index_name, skuid="2785BA"
    ):
        """
        Input:
        search_text: string
        index: string
        skuid: string
        Output:
        context: list of strings
        """
        client = self.get_search_connection(index)

        result = client.search(
            search_text,
            top=5,  # Limit to top 5 results
        )
        context = []
        for item in result:
            context.append(item["content"])
        return context

    async def summarize_context(self, context):
        prompt = (
            "Summarize the following context to provide relevant background "
            "for a cute chatbot conversation. Keep it concise, within 50 words:\n\n"
            + "\n".join(context)
        )

        client = openai.AsyncAzureOpenAI(
            # api_type=os.environ.get("OPENAI_API_TYPE"),
            azure_endpoint=os.environ.get("OPENAI_API_BASE"),
            api_version=os.environ.get("OPENAI_API_VERSION"),
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()

    def update_chat_history(self, from_phone_no, role, content):
        if from_phone_no not in self.chat_history:
            self.chat_history[from_phone_no] = []

        if len(self.chat_history[from_phone_no]) > 200:
            self.chat_history[from_phone_no].pop(0)
        self.chat_history[from_phone_no].append({"role": role, "content": content})

    async def summarize_history(self, phone_no: str, current_query: str = None) -> str:
        """
        Summarize chat history with emphasis on relevance to current query.
        
        Args:
            phone_no: User's phone number
            current_query: Current user query to determine relevance
        
        Returns:
            str: Summarized history focusing on relevant context
        """
        if phone_no not in self.chat_history or not self.chat_history[phone_no]:
            return ""

        history = self.chat_history[phone_no]
        
        # If no current query, return recent history with basic cleaning
        if not current_query:
            recent_history = history[-2000:]  # Keep last  exchanges
            return "\n".join([f"{entry['role']}: {clean_text(entry['content'])}" 
                             for entry in recent_history])

        # Calculate relevance scores for each history entry
        scored_history = []
        for i, entry in enumerate(history):
            score = calculate_similarity_score(current_query, entry)
            # Boost score for recent messages
            recency_boost = (i + 1) / len(history)
            final_score = score * 0.7 + recency_boost * 0.3
            scored_history.append((entry, final_score))

        # Sort by score and take top relevant entries
        scored_history.sort(key=lambda x: x[1], reverse=True)
        relevant_history = [entry for entry, _ in scored_history[:50]]  # Keep top 20 relevant entries, to allow chatbot to remember more 

        # Ensure chronological order for the selected entries
        relevant_history.sort(key=lambda x: history.index(x))

        # Combine relevant entries into a single string
        return "\n".join([
            f"{entry['role']}: {clean_text(entry['content'])}" 
            for entry in relevant_history
        ])

    def summarize_context_and_history(self, context, phone_no):
        """
        Summarizes both the provided context and conversation history using a single API call.

        Args:
            context (list): A list of context strings to be summarized.
            phone_no (str): The phone number identifier for fetching conversation history.

        Returns:
            str: A combined summary of context and history.
        """
        # Prepare context text
        context_text = "\n".join(context)

        # Prepare history text
        history_text = "\n".join(
            [
                f'{entry["role"]}: {entry["content"]}'
                for entry in self.chat_history[phone_no]
            ]
        )

        # Create a single prompt combining both context and history
        prompt = (
            "Summarize the following context and conversation history seperately to retain key background "
            "information for a chatbot response. Keep it concise,20 words of context, 20 words of history:\n\n"
            "summarize History:\n" + history_text + "without losing information \n\n"
            "summarize Context:\n" + context_text + "\n\n"
            "seperate Context and History by formating as json '{'History': '...','Context': '...' }"
        )

        # Call the OpenAI API once
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=100,  # Adjust as needed to fit both summaries
        )

        # Return the summarized content
        return response.choices[0].message.content.strip()

    def build_system_message(self, prompt, context, history):
        # Include summarized history and context in the system message
        system_message = {
            "role": "system",
            "content": f"{prompt}\n\nContext: {context}\n\nHistory: {history}",
        }
        return system_message
    
    def clean_user_message(self, message: list[dict]):
        """
        Replaces any messages that contain dangerous words with a placeholder.
        """
        dangerous_words = [
                            "kill", 
                            "die", 
                            "suicide", 
                            "unalive", 
                            "un alive", 
                            "end my life", 
                            "harm myself", 
                            "take my life", 
                            "end it all", 
                            "give up", 
                            "no reason to live", 
                            "can't go on", 
                            "escape this pain", 
                            "I'm done", 
                            "I just want to disappear", 
                            "I want out", 
                            "I'm broken", 
                            "tired of fighting", 
                            "I can't take it anymore", 
                            "I don't want to be here", 
                            "end the suffering", 
                            "sick of life", 
                            "I'm worthless", 
                            "I wish I was never born", 
                            "I want to fade away", 
                            "I'm a burden", 
                            "everyone would be better off without me", 
                            "I don't care anymore", 
                            "nobody would miss me", 
                            "no hope left", 
                            "I feel like giving up", 
                            "my life isn't worth living", 
                        ]
        for entry in message:
            if any(word in entry["content"].lower() for word in dangerous_words):
                entry["content"] = "[ dangerous content removed ] please help me talk through extremely negative thoughts about myself or others, ask why I am getting such thoughts, give a step by step guidance in form of counselling."
        return message
    

    def is_gibberish(self,text: str) -> bool:
        # Simple gibberish detection
        if len(text.strip()) < 2:  #or greetings like hi hello hey
            return True
        # Check if text contains at least some valid words
        words = text.split()
        if len(words) == 0:
            return True
        # Count valid words
        valid_word_count = sum(1 for word in words if word.lower() in english_words)
        if valid_word_count / len(words) < 0.5:
            return True
        # Check for excessive special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', text)) / len(text)
        return special_char_ratio > 0.5



    async def get_completion_from_messages_with_context(
        self, messages, model="gpt-4o", temperature=0.2, phone_no=None
    ):
        context = self.get_context_from_azure_search(messages[-1]["content"])

        # summarized_context_and_history = self.summarize_context_and_history(context, phone_no)
        summarized_context = await self.summarize_context(context)

        history = await self.summarize_history(phone_no, messages[-1]["content"])

        # clean user message from self harm words
        messages = self.clean_user_message(messages)




        prompt = (
            "Use the following context and history to guide your responses effectively:"
            f"\n Context: {summarized_context}"
            f"\n History: {history}"
        ) + self.prompt


        system_message = {"role": "system", "content": prompt}
        messages.insert(0, system_message)

        #save messages to a file
        with open("messages.json", "w") as f:
            json.dump(messages, f)

        # save messages to a file
        client = openai.AsyncAzureOpenAI(
            # api_type=os.environ.get("OPENAI_API_TYPE"),
            azure_endpoint=os.environ.get("OPENAI_API_BASE"),
            api_version=os.environ.get("OPENAI_API_VERSION"),
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=120,
        )

        self.update_chat_history(
            from_phone_no=phone_no,
            role="user",
            content=messages[-1]["content"],
        )
        self.update_chat_history(
            from_phone_no=phone_no,
            role="assistant",
            content=response.choices[0].message.content,
        )
        return response.choices[0].message.content

    def get_completion_from_messages_with_vector_search(
        self,
        messages,
        model="gpt-4o",
        temperature=0.15,
        from_phone_no=None,
        db=None,
        category="general",
    ):
        context = ai_search_helper.search_documents(
            query=messages[-1]["content"]
        )  # this will give us context

        # context is an iterator for now
        context = list(context)
        context = ",".join([item["content"] for item in context])[:100]

        # Ensure self.chat_history is a list of properly structured messages
        if len(self.chat_history) == 0:
            self.chat_history[from_phone_no] = []
        if isinstance(self.chat_history[from_phone_no], list):
            messages.extend(
                self.chat_history[from_phone_no]
            )  # Append previous chat history
        else:
            raise ValueError("Chat history must be a list of valid messages.")

        if context:
            # prompt = (
            #     "You are Assisti, a warm, empathetic guide, helping users explore feelings, self-reflect, "
            #     "and cope. No diagnosis, no treatment, no repetition."
            # )

            # get prompt from db, model = Prompts

            prompt = db.query(Prompts).filter_by(category=category).first().prompt

            system_message = self.build_system_message(
                prompt, context, self.chat_history[from_phone_no]
            )
            messages.append(system_message)

        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )

        # save the history in this manner
        self.update_chat_history(
            from_phone_no=from_phone_no, role="user", content=messages[-1]["content"]
        )

        self.update_chat_history(
            from_phone_no=from_phone_no,
            role="assistant",
            content=response.choices[0].message.content,
        )

        return response.choices[0].message.content

