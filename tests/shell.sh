curl http://43.180.254.145:9001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"Qwen3.5-397B-A17B-W8A8-gptq-v2\",\"messages\":[{\"role\":\"user\",\"content\":\"你好，介绍一下你自己\"}],\"max_tokens\":2048,\"temperature\":0.7}"


curl -X POST 'https://tokenhub.tencentmaas.com/v1/chat/completions' \
  -H 'Authorization: Bearer sk-v5S2Xk6dzdPgLiVTLcvjTcrjbC31VPy5yHFV9gozonn5Skfs' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "hy3",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "你好"}
    ],
    "stream": false
  }'