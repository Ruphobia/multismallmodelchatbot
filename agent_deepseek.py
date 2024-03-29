#!/usr/bin/python3
import sys
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from typing import TYPE_CHECKING, Any, Dict, List
from multiprocessing import Process, Queue
import time

input_queue = None
output_queue = None
process = None

class MyCallBack(BaseCallbackHandler):
    def __init__(self):
        self.output_queue = output_queue

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts running."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        output_queue.put(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when LLM errors."""

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when chain errors."""

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        pass

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Run when tool ends running."""

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        print("Agent Finished")

def load_llm(temperature):
    callback_manager = CallbackManager([MyCallBack()])
     
    llm = LlamaCpp(
    model_path="models/deepseek/deepseek-coder-7b-instruct-v1.5.q8_0.gguf",
    temperature=temperature,
    # n_batch=2048,
    n_gpu_layers=-1,
    callback_manager=callback_manager,
    streaming=True,
    n_ctx=4096,
    # f16_kv=True,  
    verbose=False,
    # repeat_penalty=1,
    max_tokens=4096
    )

    return llm

def get_conversation_chain(llm, conversation_memory):
    template = """You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer. Use this history to aid in your answers {history}
### Instruction:
{input}
### Response:"""
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    
    conversation = ConversationChain(
                prompt=PROMPT,
                llm=llm,
                verbose=False,
                memory=conversation_memory,
            )

    return conversation

def llmprocess(input_queue, oqueue):
    global output_queue
    output_queue = oqueue
    """
    Process user questions from an input queue.

    Input Parameters:
    input_queue: multiprocessing.Queue - A queue for receiving input.
    """
    llm = load_llm(0.1)
    conversation_memory = ConversationBufferMemory()
    while True:
        user_question = input_queue.get()  # Wait for input
        if user_question == "quit":
            print("Shutting down the process...")
            break

        Conversation_chain = get_conversation_chain(llm, conversation_memory)
        Conversation_chain.predict(input=user_question)
        
def startAgent(output_queue):
    global input_queue
    global process

    input_queue = Queue()
    process = Process(target=llmprocess, args=(input_queue,output_queue))
    process.start()


