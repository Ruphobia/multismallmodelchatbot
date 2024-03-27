


# Application imports
import agent_deepseek

global_selected_agent = None
global_agent_object = None

def start_agent(prompt, agent_task, output_queue):
    global global_selected_agent
    global global_agent_object

    if (global_selected_agent != agent_task):
        if (global_selected_agent != None):
            print("Shut down current agent")


        if (agent_task == "code"):
            agent_deepseek.startAgent(output_queue)
            global_agent_object = agent_deepseek

    global_agent_object.input_queue.put(prompt) 


