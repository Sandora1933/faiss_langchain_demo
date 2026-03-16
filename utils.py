from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from prompts import PREFIX, QUESTION_PROMPT


def build_prompt():
    """
    Build the prompt template used for the LLM.
    The prompt has three components: System message, Chat history, and User message.
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", PREFIX + "\n" + QUESTION_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{query}"),
        ]
    )
    return prompt_template