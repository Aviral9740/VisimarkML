# Plagiarism Detection Web App

This repository hosts a **plagiarism detection web application** built using **Streamlit**.  
The app compares input text against both **local data** and **live web content** to identify potential plagiarism.

---

## 🚀 Live Demo

🔗 **Demo URL:** [Demo](https://plagiarism-detector-app.streamlit.app/)

> Replace the above placeholder with your deployed Streamlit app link.

---

## ✨ Features

- **Real-Time Plagiarism Detection**: Compares user-provided text against local documents and performs live web searches.
- **Multiple Similarity Techniques**:
  - N-gram similarity  
  - TF-IDF based cosine similarity  
  - Longest Common Subsequence (LCS)
- **Live Web Integration**: Checks plagiarism against real-time web sources in addition to local data.
- **Streamlit Interface**: Clean and interactive UI for easy usage by non-technical users.

---

## ⚙️ How It Works

1. **Input Text**: User enters or uploads the text to be checked.
2. **Preprocessing**: Text is cleaned and tokenized.
3. **Similarity Analysis**:  
   - N-grams for pattern overlap  
   - TF-IDF + cosine similarity for semantic similarity  
   - LCS for sequence-level similarity
4. **Web & Local Comparison**: Input text is compared against both local data and live web results.
5. **Results**: Similarity scores and potential plagiarism indicators are displayed on the UI.

---

## 🛠️ Getting Started

### Prerequisites

- Python 3.9
- Streamlit
- Required Python packages (listed in `requirements.txt`)

### Installation

Clone the repository:

```bash
git clone https://github.com/Aviral9740/Plagiarism-Detection-Streamlit.git
cd Plagiarism-Detection-Streamlit
