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
from multiprocessing import Queue, Process

# Application Imports
from intent import IntentClassifier
from helpers import save_Prompts 
from sysmon import get_cpu_temperatures, get_gpu_temperatures, get_cpu_usage, get_gpu_usage, get_system_ram, get_gpu_ram_usage   
from agents import start_agent


pn.extension()
intentclassifier = IntentClassifier()

output_queue = Queue()
global_selected_agent = "None"

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

output_stream = pn.pane.Markdown("",
                                     sizing_mode='stretch_both',
                                     styles={'overflow-y': 'auto', 'height': '100%', 'max-height': 'calc(100vh - 100px)', 'background-color': '#34495e', 'color': '#ecf0f1'})


input_box = pn.widgets.TextAreaInput(placeholder="Type here...", sizing_mode='stretch_width', css_classes=['bk-text-area'], height=100, max_height=100)

chatbot_panel = None
cpu_usage_pane = None

def monitor_output_queue(output_queue):
    global output_stream
    """Function to monitor output queue and update the output stream."""
    while True:
        output_stream.object += output_queue.get()

def send_response(event=None):
    global input_box
    global output_stream
    global global_selected_agent

    user_message = input_box.value.lower().strip()
    save_Prompts(user_message)
    promptclass = intentclassifier.classify_prompt(user_message)
    start_agent(user_message, promptclass, output_queue)

    output_stream.object += f"{user_message}\n"
    input_box.value = '' 

    if ("[" not in promptclass):
        global_selected_agent = promptclass

def clear_output(event=None):
    global chatbot_panel
    output_stream = pn.pane.Markdown("",
                                     sizing_mode='stretch_both',
                                     styles={'overflow-y': 'auto', 'height': '100%', 'max-height': 'calc(100vh - 100px)', 'background-color': '#34495e', 'color': '#ecf0f1'})
    chatbot_panel[1] = output_stream

def stop_chat(event=None):
    global output_stream
    output_stream.object += "**Stop button pressed.**\n"

def generate_bar_graph_image(cpu_usage=[], gpu_usage=[], cpu_temp=[], gpu_temp=[], cpu_ram=[], gpu_ram=[], CurrentModel="None",
                             width=5, height=20, background_color='#34495e', bar_color='#71bf45',
                             text_color='#ffffff', section_padding=10, save_path='bar_graph_image.png'):
    """
    Generates a composite bar graph as an image with multiple metrics 
    and saves it to disk or returns it as a PIL Image object
    """
    try:
        font = ImageFont.truetype("arial.ttf", 10)
    except IOError:
        font = ImageFont.load_default()

    Model_Label = ["None"]

    all_sections = [cpu_usage, gpu_usage, cpu_temp, gpu_temp, cpu_ram, gpu_ram, Model_Label]
    labels = ['cpu usage', 'gpu usage', 'cpu temp', 'gpu temp', 'cpu ram', 'gpu ram', 'Model:          ']
    # calculate width required for labels
    label_widths = [max(font.getsize(label)[0] for label in labels) + section_padding for _ in all_sections]
    # calculate total width of all sections
    img_width = sum(len(section) * width for section in all_sections) + sum(label_widths) + 50
    img_height = height + 20

    img = Image.new('RGB', (img_width, img_height), color=background_color)
    draw = ImageDraw.Draw(img)

    def draw_section(section, x_offset, label):
        if (section == Model_Label):
            label = f"Model: {CurrentModel}"
            draw.text((x_offset, 2), label, font=font, fill=text_color)
            label_w, _ = draw.textsize(label, font=font)
            x_offset += label_w + section_padding
        else:
            draw.text((x_offset, 2), label, font=font, fill=text_color)
            
            for i, value in enumerate(section):
                bar_height = max(int((value / 100.0) * (height - 2)), 1)
                bar_top = img_height - bar_height - 2
                fill_color = bar_color
                draw.rectangle([x_offset + i * width, bar_top, x_offset + (i + 1) * width - 1, img_height - 2], fill=fill_color)

            label_w, _ = draw.textsize(label, font=font)
            x_offsetA = (len(section) * width) + section_padding
            x_offsetB = label_w + section_padding

            if (x_offsetA > x_offsetB):
                x_offset += x_offsetA
            else:
                x_offset += x_offsetB

        return x_offset

    x_offset = 0
    for section_data, label in zip(all_sections, labels):
        if section_data:
            x_offset = draw_section(section_data, x_offset, label)

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
    # Define a function that takes a PIL image as input

    # Create a new BytesIO object
    # This is a type of file object that handles bytes data
    img_byte_arr = io.BytesIO()

    # Save the PIL image into img_byte_arr
    # The image is saved as a PNG format
    # The format parameter specifies that we want to save as a PNG
    pil_img.save(img_byte_arr, format='png')

    # After saving the image into img_byte_arr
    # Return img_byte_arr which is a byte array representation of our PIL image
    return img_byte_arr

def update_cpu_usage_pane_image(new_image):
    """Update the cpu_usage_pane with new image data directly."""
    global cpu_usage_pane
    # Convert PIL image to byte array
    new_image_data = pil_image_to_byte_array(new_image)
    # Update the pane's object with the new image data
    cpu_usage_pane.object = new_image_data

def background_task():
    global cpu_usage_pane  # Declare that we're using a global variable
    global global_selected_agent  # Declare that we're using a global variable
    while True:  # Start a loop that runs indefinitely
        # Fetch new data from various system metrics
        _, cpu_usage, _ = get_cpu_usage()
        _, gpu_usage, _ = get_gpu_usage()
        _, cpu_temp, _ = get_cpu_temperatures()
        _, gpu_temp, _ = get_gpu_temperatures()
        _, cpuram, _ = get_system_ram()
        _, gpu_ram, _, _ = get_gpu_ram_usage()

        currentmodel = global_selected_agent  # Get current selected agent

        cpu_ram = [cpuram]  # Create a list with cpu ram usage

        # Generate a new image based on system metrics
        cpu_usage_img = generate_bar_graph_image(cpu_usage=cpu_usage, gpu_usage=gpu_usage, cpu_temp=cpu_temp,
                                                gpu_temp=gpu_temp, cpu_ram=cpu_ram, gpu_ram=gpu_ram,
                                                CurrentModel=currentmodel)

        # Update image on cpu usage pane
        update_cpu_usage_pane_image(cpu_usage_img)

        # Sleep for 1 second before next iteration
        time.sleep(1)


# This function is used for launching the server
def launch_server():
    # Declare chatbot_panel as a global variable
    global chatbot_panel

    # Create a chatbot
    chatbot_panel = create_chatbot()

    # Define a function that serves as a panel application
    def panel_app(doc):
        # Serve document using chatbot
        return chatbot_panel.server_doc(doc)

    # Create a background thread that runs a background task
    background_thread = threading.Thread(target=background_task, daemon=True)
    # Start the background thread
    background_thread.start()

    # Create a monitor thread that monitors output queue
    monitor_thread = threading.Thread(target=monitor_output_queue, args=(output_queue,))
    # Start monitor thread
    monitor_thread.start()

    # Create a server with specified settings
    # '/': panel_app is a route that calls panel_app function
    # io_loop=ioloop() is a loop for handling I/O operations
    # address='0.0.0.0' means server can accept connections from any IP address
    # port=5006 is a port number on which server listens
    # allow_websocket_origin=["192.168.1.164:5006"] allows connections from specified IP address
    server = Server({'/': panel_app}, io_loop=IOLoop(), address='0.0.0.0', port=5006, allow_websocket_origin=["192.168.1.164:5006"])
    # Start server
    server.start()
    # Start I/O loop for server
    server.io_loop.start()

if __name__ == '__main__':
    launch_server()
