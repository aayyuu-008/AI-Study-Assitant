# 📚 AI Study Assistant

Python (Flask) + SQLite based web app. Notes/PDF upload karo aur summary, MCQ quiz,
practice questions, aur "ask my notes" chatbot use karo.

**No API key needed** — AI ke liye [Ollama](https://ollama.com) use hota hai, jo tumhare
apne computer par free, offline chalta hai.

## 1. Ollama install karo (ek baar)

1. https://ollama.com se apne OS (Windows/Mac/Linux) ke liye Ollama download + install karo.
2. Terminal kholo aur ek chhota model pull karo:
   ```bash
   ollama pull llama3.2
   ```
   (~2GB download, ek baar hi hoga). Agar computer thoda kamzor hai to isse bhi try kar sakte ho:
   `ollama pull llama3.2:1b` (chhota, tez, thoda kam smart) — aur phir neeche step 3 mein
   `OLLAMA_MODEL` env variable us naam se set kar dena.
3. Install ke baad Ollama generally background mein apne aap chalta rehta hai (system tray/menu bar mein dikhega).
   Agar nahi chal raha to manually start karo:
   ```bash
   ollama serve
   ```

## 2. App ka local setup

```bash
cd study_assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

(Optional) agar dusra model use karna hai to:
```bash
export OLLAMA_MODEL="llama3.2:1b"     # Windows (PowerShell): $env:OLLAMA_MODEL="llama3.2:1b"
```

## 3. Run karo (localhost)

```bash
python app.py
```

Browser mein kholo: **http://127.0.0.1:5000**

Pehli baar chalane par `study_assistant.db` (SQLite file) khud-ba-khud ban jayegi.

> ⚠️ Agar summary/MCQ/chat error de raha hai, to pehle check karo Ollama chal raha hai:
> browser mein `http://localhost:11434` khol ke dekho — "Ollama is running" dikhna chahiye.

## 4. Deploy (production)

Koi bhi Python-friendly host chalega — options:

- **Render.com** — free tier, GitHub repo connect karke "Web Service" banao.
  Start command: `gunicorn app:app`
- **Railway.app** — similar, ek-click deploy.
- **PythonAnywhere** — free tier, Flask ke liye ready-made.

Deploy se pehle:
1. `requirements.txt` mein `gunicorn` add karo:  `pip install gunicorn` phir `pip freeze > requirements.txt`
2. `app.run(debug=True, ...)` line ko production mein `debug=False` kar do.
3. `study_assistant.db` ke liye persistent disk/volume use karo, warna redeploy pe data reset ho sakta hai
   — ya phir SQLite ki jagah Postgres pe switch kar lo (bade scale ke liye).
4. **Important:** Ollama free hai but sirf tumhare apne machine pe local chalta hai — free
   hosting (Render/Railway free tier) par Ollama install/run nahi ho payega (RAM/CPU limits).
   Isliye do options hain:
   - App ko sirf **apne laptop/PC par hi** chalao (localhost) — yeh sabse simple hai, deploy karne
     ki zaroorat hi nahi agar sirf khud use karna hai.
   - Sach mein online deploy karna hai to `OLLAMA_URL` ko ek aise server pe point karo jahan Ollama
     chal raha ho (apna khud ka VPS jaise DigitalOcean/Hetzner jispar Ollama install ho), ya phir
     dobara free-tier AI API (jaise Anthropic/Groq) wapas use karna padega.

## Features

| Feature | Description |
|---|---|
| Upload | PDF ya plain text notes upload karo |
| Summary | Notes ka structured summary + important topics |
| Questions | Practice short/long answer questions |
| MCQ Quiz | Auto-generated MCQs, attempt karo, turant score milega |
| Ask My Notes | Apne notes se hi sawaal-jawaab karo (RAG-style, sirf notes ke content pe based) |
| Score Tracking | Har quiz attempt DB mein save hota hai (`quiz_attempts` table) |

## Tech stack

- **Backend:** Python, Flask
- **Database:** SQLite (`study_assistant.db`)
- **AI:** [Ollama](https://ollama.com) running `llama3.2` locally — free, offline, no API key
- **Frontend:** Plain HTML/CSS/JS (no build step needed)

## Database schema (quick view)

- `notes` — uploaded notes/content
- `summaries` — generated summaries + topics
- `mcqs` — generated MCQs per note
- `quiz_attempts` — score history
- `chat_history` — "ask my notes" conversation log
