
class RAGException(Exception):
    
    pass

class RetrievalError(RAGException):
    pass

class LLMGenerationError(RAGException):
    pass