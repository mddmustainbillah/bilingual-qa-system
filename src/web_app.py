import gradio as gr
from inference import QASystem

def create_web_app(model_path):
    qa_system = QASystem(model_path)
    
    def process_question(question, language):
        return qa_system.generate_answer(question, language)
    
    # Create the interface
    iface = gr.Interface(
        fn=process_question,
        inputs=[
            gr.Textbox(label="Enter your question / আপনার প্রশ্ন লিখুন"),
            gr.Radio(
                ["English", "বাংলা"], 
                label="Select Language / ভাষা নির্বাচন করুন",
                value="English"
            )
        ],
        outputs=gr.Textbox(label="Answer / উত্তর"),
        title="Bilingual Question Answering System / দ্বিভাষিক প্রশ্ন-উত্তর সিস্টেম",
        description="Ask questions in English or Bengali / ইংরেজি বা বাংলায় প্রশ্ন করুন"
    )
    
    return iface

if __name__ == "__main__":
    app = create_web_app("./models/qa_model")
    app.launch() 