import streamlit as st
import openai
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
import os
import pinecone
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from langchain.schema.prompt_template import BasePromptTemplate

from langchain.chains.qa_with_sources import (
    map_reduce_prompt,
    refine_prompts,
    stuff_prompt,
)

import pandas as pd
import io
import random


random.seed(42)  


# Set up credentials
AZURE_OPENAI_API_KEY = st.secrets['AZURE_OPEN_AI_API_KEY']#st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_API_BASE = st.secrets['OPENAI_API_BASE']#st.secrets["AZURE_OPENAI_API_BASE"]
AZURE_OPENAI_API_TYPE = 'azure'
AZURE_OPENAI_API_VERSION = st.secrets['OPENAI_API_VERSION']
AZURE_OPENAI_EMBEDDINGS_MODEL_NAME= st.secrets['AZURE_EMBEDDINGS_DEPLOYMENT_NAME']
AZURE_OPENAI_GPT4_MODEL_NAME= st.secrets['AZURE_LLM_DEPLOYMENT_NAME']

PINECONE_INDEX_NAME=st.secrets['PINECONE_INDEX_NAME']
PINECONE_API_KEY=st.secrets['PINECONE_API_KEY']#st.secrets["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT=st.secrets['PINECONE_ENVIRONMENT']

pre_question_prompt="Answer the following question in Dutch:"

if 'search_policy' not in st.session_state:
    st.session_state['search_policy'] = 'map_reduce'

def load_prompt_given_search_policy():
    st.session_state['result_df']=None
    print('LOADING PROMPT============')
    print('previous state============')
    print(st.session_state['search_policy'])
    search_policy=st.session_state['search_policy']
    if search_policy=='refine':
      st.session_state['QUESTION_PROMPT']=map_reduce_prompt.QUESTION_PROMPT
      print(st.session_state['QUESTION_PROMPT'].dict())
      #st.session_state['COMBINE_PROMPT']=map_reduce_prompt.COMBINE_PROMPT
      #st.session_state['EXAMPLE_PROMPT']=map_reduce_prompt.EXAMPLE_PROMPT
    else:
      st.session_state['QUESTION_PROMPT']=refine_prompts.DEFAULT_TEXT_QA_PROMPT
      #st.session_state['COMBINE_PROMPT']=refine_prompts.COMBINE_PROMPT
      #st.session_state['EXAMPLE_PROMPT']=refine_prompts.EXAMPLE_PROMPT




if 'QUESTION_PROMPT' not in  st.session_state:
   load_prompt_given_search_policy()



      


default_question="Hoeveel mag ik max opgeven voor kinderopvang?"



embedder = OpenAIEmbeddings(
    openai_api_base=AZURE_OPENAI_API_BASE, 
    openai_api_key=AZURE_OPENAI_API_KEY, 
    openai_api_type=AZURE_OPENAI_API_TYPE,
    deployment=AZURE_OPENAI_EMBEDDINGS_MODEL_NAME,
    model='text-embedding-ada-002',
    chunk_size=1)


llm = AzureChatOpenAI(
    temperature=0,
    #top_p=0.0001,
    openai_api_base=AZURE_OPENAI_API_BASE, 
    openai_api_key=AZURE_OPENAI_API_KEY, 
    openai_api_version=AZURE_OPENAI_API_VERSION, 
    deployment_name=AZURE_OPENAI_GPT4_MODEL_NAME)



pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT)



@st.cache_resource
def load_pinecone_existing_index():
    pass
    docsearch = Pinecone.from_existing_index(
            index_name="qa-app",
            embedding=embedder,
            namespace="pb_v1")
    return docsearch


        

def print_documents(documents):
  pretty_string=""
  for document in documents:
    pretty_string+=("  \nSource: {}".format(document.metadata['source']))
    pretty_string+=("  \nPage: {}".format(document.metadata['page']))
    pretty_string+=("  \nContent: {}".format(document.page_content))
    pretty_string+='  \n'+'='*40
  return pretty_string

def unpack_document_objects_to_dataframe(question,question_id,answer,document_objects):
  """Unpack a list of document objects into a pandas dataframe.

  Args:
    document_objects: A list of document objects.

  Returns:
    A pandas dataframe.
  """

  df = pd.DataFrame(columns=['Question_id','Question','Document','Page','Content','Answer'])
  for document_object in document_objects:
    df_row = {
        'Document':document_object.metadata['source'],
        'Page':document_object.metadata['page'],
        'Content': document_object.page_content,
        'Question':question,
        'Question_id':question_id,
        'Answer':answer
    }
    df=pd.concat([df, pd.DataFrame([df_row])], ignore_index=True)
  return df      

def to_excel(df: pd.DataFrame):
    in_memory_fp = io.BytesIO()
    df.to_excel(in_memory_fp)
    # Write the file out to disk to demonstrate that it worked.
    in_memory_fp.seek(0, 0)
    return in_memory_fp.read()

# Streamlit app
def main():
    # Logo
    logo_image = "https://static.wixstatic.com/media/0a5b13_5016bac785d14ac9a630ae9236baeadb~mv2.png/v1/fit/w_2500,h_1330,al_c/0a5b13_5016bac785d14ac9a630ae9236baeadb~mv2.png"  # Replace with the path to the logo image
    st.image(logo_image, use_column_width=True)
    
    st.title("PB assistant bot")

    #st.subheader("upload documents to vectorDB")
    #source_doc = st.file_uploader("Upload source document", type="pdf", label_visibility="collapsed")
    
    st.session_state['search_policy'] = st.selectbox(
    'Which search policy does the llm need to use?',
    ('map_reduce', 'refine'),on_change=load_prompt_given_search_policy)
    print('new state============',st.session_state['search_policy'])

    # LLM Prompt input
    #st.subheader("LLM Prompt")
    #prompt = st.text_area("Enter your LLM prompt here:", height=350,value=st.session_state['QUESTION_PROMPT'].dict()['template'])

   
    # Questions input
    st.subheader("Questions")
    questions = st.text_area("Enter your questions here (one per line for multiple questions):", height=350,value=default_question)

    # Generate responses
    if st.button("Generate Responses"):
        #question_prompt=st.session_state['QUESTION_PROMPT']
        #print('question_prompt',question_prompt)
        if questions:
            # Split questions into a list
            question_list = questions.split("\n")
            # Generate responses for each question
            aux_dfs=[]
            for question_id,question in enumerate(question_list):
                st.write(f"Question: {question}")
                response = generate_response(pre_question_prompt+" "+question)#,question_prompt)
                st.write(f"Response: {response['answer']}")
                new_df=unpack_document_objects_to_dataframe(question,question_id,response['answer'],response['source_documents'])
                aux_dfs.append(new_df)
                with st.expander("SOURCES"):
                    st.dataframe(new_df[['Document','Page','Content']])     

            
            result_df=pd.concat(aux_dfs)
            st.session_state['result_df']=result_df
            print(result_df)


    if 'result_df' in st.session_state and st.session_state['result_df'] is not None:
        excel_data = to_excel(result_df)
        file_name = "output.xlsx"
        st.download_button(
            "Download output as Excel",
            excel_data,
            file_name)




            
            
            
        

# Generate response using OpenAI API
def generate_response(question):#,question_prompt_input):
    #print('in genarate response question_prompt',question_prompt_input)
    qa_sources_chain = load_qa_with_sources_chain(llm=llm, chain_type=st.session_state['search_policy'])#, verbose=True,question_prompt=question_prompt_input)
    vectorstore=load_pinecone_existing_index()
    qa_chain = RetrievalQAWithSourcesChain(
                combine_documents_chain=qa_sources_chain, 
                retriever=vectorstore.as_retriever(),
                return_source_documents=True
        )
    print('question',question)
    answer=qa_chain({"question": question})
    return answer


if __name__ == "__main__":
    main()
