# Event Promo Generator

Generate promotional posters and speaker videos for events using DashScope APIs (Qwen Image 2.0 Pro, HappyHorse 1.1 T2V/R2V, Qwen Plus).

## Requirements

- Python 3.6+
- A [DashScope API key](https://dashscope.console.aliyun.com/)

## How to Run

```bash
python3 server.py
```

Then open **http://localhost:8765** in your browser and enter your DashScope API key in the UI.

You can also set the key as an environment variable so it's pre-filled:

```bash
export DASHSCOPE_API_KEY=sk-...
python3 server.py
```

## Features

- **Image mode** — Describe your event, get 3 AI-suggested creative directions, generate 4 poster variants, pick a winner and refine.
- **Video mode** — Choose a speaker persona, generate a reference headshot, then produce a short promo video with lip-synced speech (5/10/15s, vertical/horizontal/square).
