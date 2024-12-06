# Bilingual Question Answering System

A sophisticated question answering system that supports both English and Bengali (বাংলা) languages. Built using TinyLlama and fine-tuned with DPO (Direct Preference Optimization) and RAG (Retrieval-Augmented Generation) for enhanced performance.

## About the Project

This system is designed to provide accurate answers to questions in both English and Bengali languages. It utilizes TinyLlama as the base model and incorporates several advanced techniques:
- Fine-tuning on custom QA datasets
- DPO training for better response alignment
- RAG implementation for context-aware answers
- Bilingual support with a user-friendly interface

## Features

- **Bilingual Support**: 
  - Full support for English and Bengali (বাংলা) questions and answers
  - Seamless language switching in the interface

- **Advanced AI Capabilities**:
  - Context-aware responses using RAG
  - Fine-tuned responses using DPO
  - Optimized for both accuracy and response time

- **User Interface**:
  - Clean and intuitive Gradio web interface
  - Easy language selection
  - Real-time response generation

## Model Capabilities

The system can handle various types of questions:
- General knowledge queries
- Factual questions
- Explanatory questions
- Simple reasoning tasks
- Context-based questions (when using RAG)

Note: The model is trained on a specific dataset and may not be suitable for:
- Complex mathematical calculations
- Real-time data queries
- Personal advice
- Medical or legal advice

## Project Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- At least 8GB RAM
- (Optional) CUDA-compatible GPU for faster inference

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/mustainbillah/bilingual-qa-system.git
   cd bilingual-qa-system
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download and prepare the model:
   ```bash
   python src/train.py  # This will download and fine-tune the model
   ```

5. Launch the web interface:
   ```bash
   python src/web_app.py
   ```

The application will be available at `http://127.0.0.1:7860`

## Usage

1. Open the web interface in your browser
2. Select your preferred language (English/বাংলা)
3. Type your question in the text box
4. Click submit and wait for the response

## Performance Considerations

- CPU Mode: The system runs in CPU mode by default but may be slower
- GPU Mode: Significantly faster if a CUDA-compatible GPU is available
- Memory Usage: Requires approximately 4-8GB RAM in CPU mode

## Limitations

- Response time may vary based on hardware capabilities
- Limited to the knowledge contained in the training data
- May not understand complex contextual nuances
- Bengali support is limited to modern standard Bengali

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TinyLlama team for the base model
- Hugging Face for transformers library
- Gradio team for the web interface framework

## Contact

Md Mustain Billah
[mustainbillah@gmail.com](mailto:mustainbillah@gmail.com)



