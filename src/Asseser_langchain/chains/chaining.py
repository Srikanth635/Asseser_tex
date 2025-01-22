from typing import List, Any
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain.chains.transform import TransformChain
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from langchain_openai.llms import OpenAI

def seq_chaining(Chains,inputs,outputs):
    sequential_chain = SequentialChain(
        chains=Chains,
        input_variables=inputs,
        output_variables=outputs,
    )
    return sequential_chain

def llm_chaining(llm, prompt, output_key):
    llm_chain = LLMChain(
        llm=llm,prompt=prompt,output_key=output_key,
    )
    return llm_chain


from langchain.chains.base import Chain

class CustomSequentialChain(Chain):
    def __init__(self, llm_client, components):
        super().__init__()
        self.llm = llm_client
        self.components = components

    @property
    def input_keys(self):
        # Define the required input keys
        return ["input_document", "initial_prompt"]

    @property
    def output_keys(self):
        # Define the output keys based on the number of steps
        return [f"output_step_{i + 1}" for i in range(len(self.components))]

    def _call(self, inputs):
        doc = inputs["input_document"]
        prev_output = None
        results = {}

        for i, step in enumerate(self.components):
            prompt_template = step
            # Generate the formatted prompt
            if i == 0:
                # First step uses initial prompt
                prompt = prompt_template.format(document=doc, prev_output="")
            else:
                # Subsequent steps use previous output
                prompt = prompt_template.format(document=doc, prev_output=prev_output)

            # Call the LLM with the prompt
            output = self.llm(prompt)
            results[f"output_step_{i + 1}"] = output
            prev_output = output  # Set the previous output for the next step

        return results