import httpx

prompt = "공주대학교의 수시 모집 일정은 어떻게 되나요?"

response = httpx.post(
    'http://localhost:11434/api/generate',
    json={
        'model': 'mistral',
        'prompt': prompt,
        'stream': False
    },
    timeout=120
)

print(response.json()['response'])
