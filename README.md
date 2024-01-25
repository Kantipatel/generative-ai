# Respository: generative-ai

Main repository for hosting Python code related to Generative AI code that integrates and communicates with custom LLMs as well as known third-party providers of API services that host third-party LLMs.

This repository's main code base is Python.

#OpenAI Setup

```
pip install openai
```
Authentication
The OpenAI API uses API keys for authentication. Visit your API Keys page to retrieve the API key you'll use in your requests.

All API requests should include your API key in an Authorization HTTP header as follows:
```
Authorization: Bearer OPENAI_API_KEY
```

Making API request to OpenAI:
```
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```

You should get a response back that resembles the following:
```
{
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1677858242,
    "model": "gpt-3.5-turbo-1106",
    "usage": {
        "prompt_tokens": 13,
        "completion_tokens": 7,
        "total_tokens": 20
    },
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "\n\nThis is a test!"
            },
            "logprobs": null,
            "finish_reason": "stop",
            "index": 0
        }
    ]
}
```
