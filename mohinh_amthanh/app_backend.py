import os
from flask import Flask, render_template, request
import whisper
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS

# ===== Load model =====
speech_model = whisper.load_model("base")

model_name = "Helsinki-NLP/opus-mt-en-vi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
translator = MarianMTModel.from_pretrained(model_name)

app = Flask(__name__)

# ===== Hàm chia nhỏ =====
def split_text(text, max_len=200):
    words = text.split()
    return [" ".join(words[i:i+max_len]) for i in range(0, len(words), max_len)]

# ===== Route =====
@app.route("/", methods=["GET", "POST"])
def index():
    translated_text = ""
    
    if request.method == "POST":
        file = request.files["audio"]
        path = "temp.mp3"
        file.save(path)

        # Speech to text
        result = speech_model.transcribe(path)
        text = result["text"]

        # Translate
        chunks = split_text(text)
        translated_text = ""

        for chunk in chunks:
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
            output = translator.generate(**inputs)
            translated_text += tokenizer.decode(output[0], skip_special_tokens=True) + " "

        # Text to speech
        tts = gTTS(translated_text, lang="vi")
        tts.save("static/output.mp3")

    return render_template("index.html", result=translated_text)

if __name__ == "__main__":
    app.run(debug=True)
