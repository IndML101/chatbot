## T5 based custom chatbot

This repository contains a chat bot based on T5 large language model, fine tuned on Franz Kafka's short story The Metamorphosis. Clone the repo install required dependencies and run `python app.py` to train and serve the model on `127.0.0.1:7860`. If you don't have a GPU you can also use google colab. Once the chat bot is up and running you can ask questions like:
* What is the name of the protagonist?
* Can you summarise the story?

### Dependencies
* langchain 
* transformers
* sentence_transformers
* llama-index
* gradio
* PyPDF2

### Folder Structure
`
├── chat_flan.ipynb
├── dialogGPT
│   ├── app.py
│   ├── docs
│   │   └── Franz_Kafka_The_Metamorphosis.pdf
│   ├── flagged
│   │   └── log.csv
│   └── index.json
└── readme.md
`