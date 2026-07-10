# 🌾 AgriSense AI
### Agentic AI Powered Crop Disease Diagnosis & Decision Support Platform

AgriSense AI is an intelligent multi-agent agricultural decision support system that detects crop diseases from images and provides evidence-based treatment recommendations along with economic impact analysis.

Developed as part of the **Lenovo LEAP × BharatCares × AICTE Internship Program**.

---

# Features

- 🌱 Crop disease detection from images
- 🤖 Agentic AI workflow using LangGraph
- 👁️ Vision AI for disease identification
- 📚 RAG-based evidence retrieval using FAISS
- 💊 Treatment recommendation generation
- 💰 Economic analysis with ROI estimation
- 📄 Professional PDF report generation
- 💾 Analysis history using SQLite

---

# Supported Crops

- Tomato
- Potato
- Rice
- Wheat
- Cotton
- Chili
- Pepper
- Maize
- Cotton

---

# Multi-Agent Workflow

```text
Image Upload
      │
      ▼
Vision Agent
      │
      ▼
Critic Agent
      │
      ▼
RAG Agent
      │
      ▼
Treatment Agent
      │
      ▼
Economic Agent
      │
      ▼
Planner Agent
      │
      ▼
Synthesizer Agent
      │
      ▼
PDF Report
```

---

# Technology Stack

## Frontend
- Streamlit

## Backend
- Python
- LangGraph
- LangChain
- OpenRouter
- OpenAI SDK
- Sentence Transformers

## AI
- Qwen Vision Models
- RAG
- FAISS Vector Database

## Database
- SQLite

## PDF Generation
- ReportLab

---

# System Architecture

- Vision AI detects crop disease
- Critic Agent validates diagnosis
- RAG Agent retrieves verified agricultural evidence
- Treatment Agent recommends appropriate treatment
- Economic Agent estimates treatment cost and ROI
- Planner Agent coordinates workflow
- Synthesizer Agent generates farmer-ready PDF reports

---

# Project Screenshots

## Homepage

<img width="1471" height="838" alt="image" src="https://github.com/user-attachments/assets/ef34b24e-8c4f-4b2a-bce6-6b04a61f0f70" />


---

## User Input

<img width="1469" height="838" alt="image" src="https://github.com/user-attachments/assets/9364e0e1-6a28-410f-9139-65dc931b4c96" />


---

## Analysis Dashboard

<img width="1232" height="938" alt="image" src="https://github.com/user-attachments/assets/16988294-2930-43bf-ae13-c127a56339e5" />


---

## Generated PDF Report

<img width="650" height="919" alt="image" src="https://github.com/user-attachments/assets/0419f3d3-1825-47eb-a790-10a371c4fe1f" />


---

# Installation

Clone the repository

```bash
git clone https://github.com/GaurangiSonar/AgriSense-AI.git
```

Go inside the project

```bash
cd AgriSense-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file and add your OpenRouter API key

```env
OPENROUTER_API_KEY=YOUR_API_KEY
```

Run the application

```bash
streamlit run app.py
```

---

# Future Scope

- Integrate deep learning models for higher disease detection accuracy.
- Support additional crops and plant diseases.
- Enable multilingual farmer support.
- Deploy as a mobile application.
- Integrate real-time weather and market price data.

---

# Project Outcomes

- Faster crop disease diagnosis
- Evidence-based treatment recommendations
- Economic decision support for farmers
- Professional PDF report generation
- Multi-agent AI powered agricultural assistance

---

# Developed For

Lenovo LEAP Internship

Powered by

- Lenovo
- BharatCares
- AICTE

---

# Author

**Gaurangi Sonar**

---

## License

This project is developed for educational and internship purposes.
