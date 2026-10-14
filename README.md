readme_content = """# 🛡️ PromptShield: Real-Time LLM Prompt Firewall

An interactive neural guardrail for Large Language Models designed to intercept and mitigate **OWASP LLM01: Prompt Injection & Adversarial Jailbreaks** at the preprocessing ingress tier.

## 🚀 Key Features
- **Sub-20ms Latency:** Low-overhead transformer inference running on PyTorch and CUDA.
- **Asynchronous Live Keystroke Scanning:** Real-time character-level threat evaluation built using Gradio.
- **Three-Tier Policy Triage:** Automated categorization into Safe (<40%), Suspicious (40-69%), and Critical Block (≥70%).
- **Contextual Attention:** Detects adversarial intent and persona overrides where standard regex/keyword filters fail.

## 🛠️ Tech Stack
- **Framework:** PyTorch, Hugging Face Transformers
- **Model:** DeBERTa-v3 / DistilBERT sequence classification
- **Interface:** Gradio (`gr.Blocks` with `.change` listeners)
- **Dataset:** `deepset/prompt-injections`

## 📦 Quickstart
```bash
pip install -r requirements.txt
python app.py
