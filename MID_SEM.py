import tkinter as tk #for GUI
from tkinter import ttk, messagebox #for GUI

import os #File handling for saving audio
import tempfile #For creating temporary files

from tkinter.ttk import Progressbar #For progress bar 
  
from gtts import gTTS #For text to speech

import numpy as np #For numerical operations
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Tokenizer #For speech recognition
import torch #For deep learning
import sounddevice as sd #For recording audio
import random #Used for generating random sentences
import time #Used for updating progress bar
import hashlib 
import soundfile as sf #For saving audio
import pygame  #For playing audio


class PronunciationDetector:
    def __init__(self, root, username="User"):
        self.root = root
        self.username = username
        self.root.title("Pronunciation Assistant - Mastering the Spoken Word")
        self.root.geometry("1000x800")
        self.root.configure(bg='#e0f7fa')

       # Load Wav2Vec 2.0 processor and model
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")

        # Initialize user data
        self.current_sentence = ""
        self.logged_in = True
        self.user_data = {"correct_count": 0, "incorrect_count": 0}

        # Notebook for tabbed UI
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both")

        self.create_main_screen()

    def create_main_screen(self):
        main_tab = ttk.Frame(self.notebook)
        self.notebook.add(main_tab, text="Pronunciation")

        # Header
        header_frame = tk.Frame(main_tab, bg='#00796b', height=80)
        header_frame.pack(fill=tk.X)
        self.label = tk.Label(header_frame, text=f"Welcome, {self.username}!", font=("Arial", 20, "bold"), bg='#00796b', fg='white')
        self.label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Body
        body_frame = tk.Frame(main_tab, bg='#e0f7fa')
        body_frame.pack(pady=30)

        # Sentence Input Section
        input_frame = tk.Frame(body_frame, bg='#b2dfdb', padx=20, pady=20)
        input_frame.pack(pady=10)

        sentence_label = tk.Label(input_frame, text="Enter or Get a Random Sentence:", font=("Arial", 14), bg='#b2dfdb')
        sentence_label.grid(row=0, column=0, columnspan=2, pady=10)

        self.custom_sentence_entry = tk.Entry(input_frame, font=("Arial", 14), width=50)
        self.custom_sentence_entry.grid(row=1, column=0, columnspan=2, pady=10)

        self.random_sentence_button = tk.Button(input_frame, text="Get Random Sentence", command=self.get_random_sentence, font=("Arial", 12), bg="#00796b", fg="white")
        self.random_sentence_button.grid(row=2, column=0, padx=10, pady=10)

        self.type_sentence_button = tk.Button(input_frame, text="Type Your Sentence", command=self.type_sentence, font=("Arial", 12), bg="#00796b", fg="white")
        self.type_sentence_button.grid(row=2, column=1, padx=10, pady=10)

        # Pronunciation Recording Section
        record_frame = tk.Frame(body_frame, bg='#b2dfdb', padx=20, pady=20)
        record_frame.pack(pady=10)

        record_label = tk.Label(record_frame, text="Record Your Pronunciation:", font=("Arial", 14), bg='#b2dfdb')
        record_label.grid(row=0, column=0, columnspan=2, pady=10)

        self.record_button = tk.Button(record_frame, text="Record Pronunciation", command=self.record_sentence, font=("Arial", 14), bg="#00796b", fg="white")
        self.record_button.grid(row=1, column=0, padx=10, pady=10)

        self.progress = Progressbar(record_frame, length=200, mode='determinate')
        self.progress.grid(row=1, column=1, padx=10, pady=10)

        # Results Display
        result_frame = tk.Frame(body_frame, bg='#b2dfdb', padx=20, pady=20)
        result_frame.pack(pady=10)

        self.result_text = tk.Text(result_frame, height=6, width=65, bg='#ffffff', fg='black', font=("Arial", 12), relief=tk.SUNKEN)
        self.result_text.pack(pady=10)

    def get_random_sentence(self):
        try:
            with open("database.txt", "r") as file:
                sentences = file.readlines()
            self.current_sentence = random.choice(sentences).strip()
        except FileNotFoundError:
            messagebox.showwarning("Error", "Sentences file not found.")
            return
        self.label.config(text=f"Random Sentence: {self.current_sentence}")

    def type_sentence(self):
        self.current_sentence = self.custom_sentence_entry.get()
        if not self.current_sentence.strip():
            messagebox.showwarning("Input Error", "Please enter a valid sentence.")
        else:
            self.label.config(text=f"Typed Sentence: {self.current_sentence}")

    def normalize_text(self, text):
        return ''.join(e for e in text.lower() if e.isalnum() or e.isspace()).strip()
    def speak_correct_pronunciation(self):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_path = temp_file.name
            tts = gTTS(self.current_sentence)
            tts.save(temp_path)
            pygame.mixer.init()
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            os.remove(temp_path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not play the sound: {e}")

    def record_sentence(self):
        if not self.current_sentence:
            messagebox.showwarning("Sentence Required", "Please provide a sentence first.")
            return

        self.label.config(text="Getting ready for listening...")
        fs = 16000  
        duration = 7  
        self.update_progress(duration)
        self.label.config(text=f"Sentence: {self.current_sentence}\nListening...")

        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()

        recording = np.squeeze(recording)
        input_values = self.processor(recording, sampling_rate=fs, return_tensors="pt").input_values

        self.label.config(text="Processing...")

        with torch.no_grad():
            logits = self.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        text = self.tokenizer.batch_decode(predicted_ids)[0]

        transcribed_text = self.normalize_text(text)
        expected_text = self.normalize_text(self.current_sentence)

        if transcribed_text == expected_text:
            result = "Correct pronunciation!"
            self.user_data["correct_count"] += 1
        else:
            result = "Incorrect pronunciation."
            self.user_data["incorrect_count"] += 1
            self.speak_correct_pronunciation()

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Transcribed Text:\n{text}\n{result}")

    def update_progress(self, duration):
        for i in range(101):
            self.progress['value'] = i
            self.root.update_idletasks()
            time.sleep(duration / 100)

if __name__ == "__main__":
    root = tk.Tk()
    app = PronunciationDetector(root)
    root.mainloop()
