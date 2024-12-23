from codegrep.config import OPENAI_EMBEDDING_MODEL
from langchain_openai import OpenAIEmbeddings


embedding_model = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
