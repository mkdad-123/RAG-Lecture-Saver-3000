from src.generation.llm_client import LLMClient

llm = LLMClient()

answer = llm.generate(
    question="ما هي  gradient descent?",
    context_chunks=[
        "Gradient descent is an optimization algorithm that follows the negative gradient of a function to find its local minimum.",
        "It is widely used in machine learning to update model parameters and minimize the cost function.",
        "The learning rate determines the size of the steps taken towards the minimum."
    ]
)

print(answer)

