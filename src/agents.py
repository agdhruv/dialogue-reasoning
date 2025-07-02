from dataclasses import dataclass
from openai import AzureOpenAI, OpenAI
from src.clients import get_azure_openai_client
from openai.types.chat import ChatCompletion
from typing import Callable

@dataclass
class Message:
    speaker: str
    receiver: str
    content: str
    raw_response: ChatCompletion | None

    def to_dict(self):
        return {
            "speaker": self.speaker,
            "receiver": self.receiver,
            "content": self.content,
            "raw_response": None if self.raw_response is None else self.raw_response.model_dump()
        }
    
    def __str__(self):
        return f"{self.speaker} -> {self.receiver}: {self.content}"

class Conversation:
    def __init__(self):
        self.history: list[Message] = []
    
    def add_message(self, message: Message):
        self.history.append(message)
    
    def print(self):
        for message in self.history:
            print(f"\033[93m{message.speaker} --> {message.receiver}\033[0m: {message.content}")
    
    def __str__(self):
        s = ""
        for message in self.history:
            s += f"{message.speaker} --> {message.receiver}: {message.content}\n"
        return s

    def to_list(self):
        return [message.to_dict() for message in self.history]

class Agent:
    def __init__(self, name: str, system_message: str, model_name: str, client: OpenAI | AzureOpenAI = get_azure_openai_client(), temperature: float = 0):
        '''
        Args:
            name: The name of the agent.
            system_message: The system message for the agent.
            model_name: The name of the model to use.
            client: The client to use.
            temperature: The temperature for the model.
        '''
        self.model_name = model_name
        self.client = client
        self.name = name
        self.system_message = system_message
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_message},
        ]
        self.temperature = temperature
    
    def _generate_response(self) -> ChatCompletion:
        '''Produces a response to the message history and returns the full response object.'''
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
    
    def speak(self) -> ChatCompletion:
        '''Generates a response and appends it to its own message history.'''
        response = self._generate_response()
        reply_content = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply_content})
        return response

    def listen(self, message: str):
        '''Receives a message and appends it to its own message history.'''
        self.messages.append({"role": "user", "content": message})

class Dialogue:
    def __init__(self, agent1: Agent, agent2: Agent):
        self.agent1 = agent1
        self.agent2 = agent2
        self.conversation = Conversation()

    def run(self, initial_message: str, max_turns: int = 5, terminate_function: Callable[[str], bool] = lambda reply: 'TERMINATE' in reply):
        '''Initiates and runs a conversation between two agents.'''
        
        # Agent1 starts the conversation
        speaker = self.agent1
        listener = self.agent2
        
        # Manually add the first message to the speaker's history and send it to the listener
        speaker.messages.append({"role": "assistant", "content": initial_message})
        listener.listen(initial_message)
        self.conversation.add_message(Message(speaker.name, listener.name, initial_message, None))
        
        # The listener now becomes the speaker
        speaker, listener = listener, speaker
        
        # Run the conversation for max_turns
        for _ in range(max_turns - 1):
            response = speaker.speak()
            reply_content = response.choices[0].message.content
            
            self.conversation.add_message(Message(speaker.name, listener.name, reply_content, response))
            
            if terminate_function(reply_content):
                break
            
            listener.listen(reply_content)
            
            # Swap roles for the next turn
            speaker, listener = listener, speaker

        return self.conversation