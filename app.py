#!/usr/bin/python3
from bokeh.server.server import Server
from bokeh.layouts import row
import io
from panel.pane import PNG
from PIL import Image, ImageDraw, ImageFont
from tornado.ioloop import IOLoop
import panel as pn
import threading
import time

# Application Imports
from intent import IntentClassifier
from helpers import save_Prompts 
from sysmon import get_cpu_temperatures, get_gpu_temperatures, get_cpu_usage, get_gpu_usage, get_system_ram, get_gpu_ram_usage   
from agents import output_stream, start_agent

pn.extension()
intentclassifier = IntentClassifier()

dark_theme_css = """
body {
    background-color: #2c3e50; /* Dark theme background color */
    color: #ecf0f1; /* Light text color for contrast */
    margin: 0; /* Remove default margin */
    padding: 0; /* Remove default padding */
    height: 100vh; /* Full viewport height */
    overflow: hidden; /* Hide overflow to prevent outer scrollbar */
}

.bk-root .bk-btn-default {
    background-color: #34495e; /* Darker element background */
    border-color: #445a6f;
    color: #ecf0f1; /* Light text color for contrast */
}

body .bk-root .bk-input.bk-text-area,
.bk-root .bk-text-area {
    background-color: #34495e; /* Darker element background */
    color: #ecf0f1; /* Light text color for contrast */
    border-color: #445a6f;
    resize: none !important;
}
"""
# Inject custom CSS
pn.config.raw_css.append(dark_theme_css)

# Adjustments to TextAreaInput height and output_stream max-height might be needed
input_box = pn.widgets.TextAreaInput(placeholder="Type here...", sizing_mode='stretch_width', css_classes=['bk-text-area'], height=100, max_height=100)

chatbot_panel = None
cpu_usage_pane = None

def send_response(event=None):
    global output_stream
    global input_box
    user_message = input_box.value.lower().strip()
    save_Prompts(user_message)
    promptclass = intentclassifier.classify_prompt(user_message)
    start_agent(user_message, promptclass)

    output_stream.object += f"{promptclass}\n"
    input_box.value = '' 

def clear_output(event=None):
    global chatbot_panel, output_stream
    output_stream = pn.pane.Markdown("",
                                     sizing_mode='stretch_both',
                                     styles={'overflow-y': 'auto', 'height': '100%', 'max-height': 'calc(100vh - 100px)', 'background-color': '#34495e', 'color': '#ecf0f1'})
    chatbot_panel[1] = output_stream

def stop_chat(event=None):
    global output_stream
    output_stream.object += "**Stop button pressed.**\n"

def generate_bar_graph_image(cpu_usage=[], gpu_usage=[], cpu_temp=[], gpu_temp=[], cpu_ram=[], gpu_ram=[],
                             width=5, height=20, background_color='#34495e', bar_color='#71bf45',
                             text_color='#ffffff', section_padding=10, save_path='bar_graph_image.png'):
    """
    Generates a composite bar graph as an image with multiple metrics and saves it to disk.
    """
    try:
        font = ImageFont.truetype("arial.ttf", 10)
    except IOError:
        font = ImageFont.load_default()

    all_sections = [cpu_usage, gpu_usage, cpu_temp, gpu_temp, cpu_ram, gpu_ram]
    non_empty_sections = sum(bool(section) for section in all_sections)
    img_width = (sum(len(section) * width for section in all_sections) + (non_empty_sections - 1) * section_padding if non_empty_sections else 1) + 50

    img_height = height + 20  
    img = Image.new('RGB', (max(img_width, 1), img_height), color=background_color)
    draw = ImageDraw.Draw(img)

    def draw_section(section, x_offset, label):
        draw.text((x_offset, 2), label, font=font, fill=text_color)
        for i, value in enumerate(section):
            bar_height = max(int((value / 100.0) * (height - 2)), 1)
            bar_top = img_height - bar_height - 2  
            if value <= 65:
                fill_color = 'green'
            elif value <= 75:
                fill_color = 'yellow'
            else:
                fill_color = 'red'
            draw.rectangle([x_offset + i * width, bar_top, x_offset + (i + 1) * width - 1, img_height - 2], fill=fill_color)
        return x_offset + len(section) * width + section_padding

    x_offset = 0
    if cpu_usage:
        x_offset = draw_section(cpu_usage, x_offset, 'CPU Usage')
    if gpu_usage:
        x_offset = draw_section(gpu_usage, x_offset, 'GPU Usage')
    if cpu_temp:
        x_offset = draw_section(cpu_temp, x_offset, 'CPU Temp')
    if gpu_temp:
        x_offset = draw_section(gpu_temp, x_offset, 'GPU Temp')
    if cpu_ram:
        x_offset = draw_section(cpu_ram, x_offset, 'CPU RAM')
    if gpu_ram:
        x_offset = draw_section(gpu_ram, x_offset, 'GPU RAM')

    img.save(save_path)

    return img

def pil_image_to_panel(pil_img):
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='PNG')
    return PNG(img_byte_arr.getvalue(), sizing_mode='fixed')

def create_chatbot() -> pn.Column:
    global cpu_usage_pane
    global input_box
    global output_stream
    cpu_usage_img = generate_bar_graph_image()

    cpu_usage_pane = pil_image_to_panel(cpu_usage_img)

    header = pn.Row(cpu_usage_pane, styles={'background': '#34495e'}, sizing_mode='stretch_width')

    send_button = pn.widgets.Button(name="Send", button_type="success", css_classes=['bk-btn-default'])
    send_button.on_click(send_response)
    stop_button = pn.widgets.Button(name="Stop", button_type="danger", css_classes=['bk-btn-default'])
    stop_button.on_click(stop_chat)
    clear_button = pn.widgets.Button(name="Clear", button_type="warning", css_classes=['bk-btn-default'])
    clear_button.on_click(clear_output)
    
    chat_controls = pn.Row(input_box, send_button, stop_button, clear_button, sizing_mode='stretch_width')
   
    layout = pn.Column(header, output_stream, chat_controls, sizing_mode='stretch_both', height_policy='max')

    return layout

def pil_image_to_byte_array(pil_img):
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format='PNG')
    return img_byte_arr

def update_cpu_usage_pane_image(new_image):
    """Update the cpu_usage_pane with new image data directly."""
    global cpu_usage_pane
    # Convert PIL image to byte array
    new_image_data = pil_image_to_byte_array(new_image)
    # Update the pane's object with the new image data
    cpu_usage_pane.object = new_image_data

def background_task():
    global cpu_usage_pane  # Make sure to use the global reference
    while True:
        # Fetch new data
        _, cpu_usage, _ = get_cpu_usage()
        _, gpu_usage, _ = get_gpu_usage()
        _, cpu_temp, _ = get_cpu_temperatures()
        _, gpu_temp, _ = get_gpu_temperatures()
        _, cpuram, _ = get_system_ram()
        _, gpu_ram, _, _ = get_gpu_ram_usage()

        cpu_ram = [cpuram]
        # Generate the new image
        cpu_usage_img = generate_bar_graph_image(cpu_usage=cpu_usage, gpu_usage=gpu_usage, cpu_temp=cpu_temp,
                                                 gpu_temp=gpu_temp, cpu_ram=cpu_ram, gpu_ram=gpu_ram)

        update_cpu_usage_pane_image(cpu_usage_img)

        time.sleep(1)

def launch_server():
    global chatbot_panel

    chatbot_panel = create_chatbot()

    def panel_app(doc):
        return chatbot_panel.server_doc(doc)
    
    background_thread = threading.Thread(target=background_task, daemon=True)
    background_thread.start()

    server = Server({'/': panel_app}, io_loop=IOLoop(), address='0.0.0.0', port=5006, allow_websocket_origin=["192.168.1.223:5006"])
    server.start()
    server.io_loop.start()

if __name__ == '__main__':
    launch_server()
