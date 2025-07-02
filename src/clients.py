from openai import AzureOpenAI, OpenAI
from azure.identity import ChainedTokenCredential, AzureCliCredential, ManagedIdentityCredential, get_bearer_token_provider

INSTANCE = 'gcr/shared'
API_VERSION = '2025-04-01-preview'
SCOPE = "api://trapi/.default"

def get_azure_openai_client() -> AzureOpenAI:
    """
    Create and return an Azure OpenAI client with authentication.
    
    Args:
        None
        
    Returns:
        AzureOpenAI: Configured Azure OpenAI client
    """
    # Authenticate by trying az login first, then a managed identity, if one exists on the system
    credential = get_bearer_token_provider(ChainedTokenCredential(
        AzureCliCredential(),
        ManagedIdentityCredential(),
    ), SCOPE)

    endpoint = f'https://trapi.research.microsoft.com/{INSTANCE}'

    # Create an AzureOpenAI Client
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=credential,
        api_version=API_VERSION,
    )
    
    return client

def get_openai_client() -> OpenAI:
    """
    Create and return an OpenAI client.
    """
    client = OpenAI()
    return client

def get_ollama_client(base_url: str = "http://localhost:11434/v1") -> OpenAI:
    """
    Create and return an OpenAI client for a local Ollama server.
    
    Args:
        base_url (str): The base URL of the Ollama server.
        
    Returns:
        OpenAI: A configured OpenAI client pointing to the Ollama server.
    """
    client = OpenAI(
        base_url=base_url,
        api_key='ollama', # required, but unused
    )
    return client