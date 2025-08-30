# Immigration AI - Visa Guardian

An intelligent AI-powered immigration assistant that provides accurate information about various visa types including F1, F2, H1B, H4, J1, and J2 visas.

## 🚀 Features

- **AI-Powered Chatbot**: Get instant answers to immigration-related questions
- **Comprehensive Visa Coverage**: Support for F1, F2, H1B, H4, J1, and J2 visas
- **RAG (Retrieval-Augmented Generation)**: Accurate responses based on official immigration documents
- **Modern Web Interface**: React-based frontend with TypeScript
- **FastAPI Backend**: High-performance Python API

## 🏗️ Project Structure

```
Immigration AI/
├── frontend/                 # React TypeScript frontend
│   ├── src/
│   ├── public/
│   └── package.json
├── visa_guardian/           # Python FastAPI backend
│   ├── app/
│   ├── data/
│   ├── scripts/
│   └── requirements.txt
└── README.md
```

## 🛠️ Technology Stack

### Frontend
- **React 19** with TypeScript
- **Vite** for build tooling
- **ESLint** for code quality

### Backend
- **FastAPI** for API development
- **FAISS** for vector similarity search
- **Sentence Transformers** for embeddings
- **BeautifulSoup4** for web scraping
- **Uvicorn** for ASGI server

## 📦 Installation

### Prerequisites
- Node.js (v18 or higher)
- Python (v3.8 or higher)
- Git

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Backend Setup
```bash
cd visa_guardian
pip install -r requirements.txt
python start_chatbot.py
```

## 🚀 Usage

1. Start the backend server
2. Start the frontend development server
3. Open your browser and navigate to the frontend URL
4. Ask questions about immigration and visa-related topics

## 📚 Data Sources

The system uses data from:
- Official USCIS documents
- Code of Federal Regulations (CFR)
- State Department guidelines
- BridgeUSA programs
- ICE SEVIS information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is for informational purposes only and should not be considered as legal advice. Always consult with a qualified immigration attorney for legal matters.

## 🔗 Links

- [USCIS Official Website](https://www.uscis.gov/)
- [State Department Visa Information](https://travel.state.gov/content/travel/en/us-visas.html)
- [Code of Federal Regulations](https://www.ecfr.gov/)
