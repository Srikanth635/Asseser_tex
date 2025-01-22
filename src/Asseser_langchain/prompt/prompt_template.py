from langchain.prompts import PromptTemplate

initial_template = "Assess the document based on the objectives and criteria given as \n\n {prompt} on the document with contents as follows: \n\n {document}"
chain_template = ("Assess the document based on the objectives and criteria given as \n\n {prompt} on the document with contents as follows: \n\n "
                  "{document} \n\n also consider previous output \n\n {previous_output}")

initial_prompt_template = PromptTemplate(
    template=initial_template,
    input_variables=['prompt1','document']
)

chain_prompt_template1 = PromptTemplate(
    template=chain_template,
    input_variables=['prompt2','document' ,'previous_output1']
)

# prompts = [
#     PromptTemplate(
#         input_variables=["document", "prev_output"],
#         template="Analyze the document on the objectives and criteria: \n\n{document} \n\n Previous Output:\n\n {prev_output} \n\n Step 1 Analysis:"
#     ),
#     PromptTemplate(
#         input_variables=["document", "prev_output"],
#         template="Review the document again:\n\n {document} \n\n Considering Step 1 Analysis:\n\n {prev_output} \n\n Step 2 Review:"
#     ),
#     PromptTemplate(
#         input_variables=["document", "prev_output"],
#         template="Summarize key insights: \n\n {document} \n\n Based on Step 2 Review: \n\n {prev_output} \n\n Step 3 Summary:"
#     ),
#     PromptTemplate(
#         input_variables=["document", "prev_output"],
#         template="Generate final recommendations: \n\n {document} \n\n Based on Step 3 Summary: \n\n {prev_output} \n\n Recommendations:"
#     ),
# ]

def prompt_template_generator():
    prop_template = PromptTemplate()
