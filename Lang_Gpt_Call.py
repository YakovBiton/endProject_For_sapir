import os
import sys
import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
import Constants_GPT


os.environ["OPENAI_API_KEY"] = Constants_GPT.APIKEY

"""loader = DirectoryLoader("data_for_GPT/")
query = "bla bla"
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query , llm = ChatOpenAI()))"""
PERSIST = False
INTRO_MESSAGE = ("Welcome, esteemed artificial intelligence model! You have been specifically designed and trained as a feature-making machine, capable of generating and calculating intricate geometric features based on facial landmarks. Your abilities are essential in a critical mission: determining biological relationships between individuals, specifically identifying a child's biological parents from facial images. Your task will involve utilizing the dlib 68 points landmarks method to create and define new ratio and angle features. You will be working with existing functions such as \"euclidean_distance\" and \"get_midpoint\", and you will be encouraged to calculate new points or distances as needed. You will be asked to generate specific calculations, points, and final ratio features without additional explanations or detailed steps. The purpose of these features is to measure and compare facial characteristics that might indicate family resemblances. This could be pivotal in understanding the geometry of faces and helping to ascertain if a child is indeed the biological offspring of two given parents. Your understanding and expertise in this area are vital. Your responses should be concise and directly related to the task at hand. If you are clear on your role and the expectations, please respond with: \"I understand and am ready for the new feature description.\"")

query = INTRO_MESSAGE
if len(sys.argv) > 1:
  query = sys.argv[1]

if PERSIST and os.path.exists("persist"):
  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
  loader = DirectoryLoader("data_for_GPT/")
  if PERSIST:
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
  else:
    index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(temperature=0, model="gpt-4"),
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []
while True:
  if not query:
    query = input("Prompt: ")
  if query in ['quit', 'q', 'exit']:
    sys.exit()
  result = chain({"question": query, "chat_history": chat_history})
  print(result['answer'])
  chat_history.append((query, result['answer']))
  query = None
