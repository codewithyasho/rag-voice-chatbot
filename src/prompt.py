from langchain_core.prompts import ChatPromptTemplate


# GENRAL PURPOSE PROMPT
def general_purpose_prompt():
    return ChatPromptTemplate.from_template(
        """
        You are a helpful and factual AI assistant.
        Use the following retrieved context to answer the user's question.
        If the answer is not found in the context, then reply with your knowledge.
        Be concise and to the point.

        <context>
        {context}
        </context>

        Question: {input}
    """
    )
