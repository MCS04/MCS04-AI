import gradio as gr
import random as rand

# Put ML Model here
def classify(name):
    grades = ['A', 'B', 'C']
    return "This is Grade " + grades[rand.randint(0,2)]

demo = gr.Interface(fn=classify, inputs=gr.Image(shape=(224, 224)), outputs=gr.Label(num_top_classes=3))
    
demo.launch()   