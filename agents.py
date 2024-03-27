import panel as pn


global_agents = {}
global_selected_agent = None

output_stream = pn.pane.Markdown("",
                                     sizing_mode='stretch_both',
                                     styles={'overflow-y': 'auto', 'height': '100%', 'max-height': 'calc(100vh - 100px)', 'background-color': '#34495e', 'color': '#ecf0f1'})


def start_agent(prompt, agent_task):
    global output_stream
    output_stream.object += f"{prompt}, {agent_task}"

