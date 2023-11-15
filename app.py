# Bring in deps
import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ['OPENAI_API_KEY'] = apikey

#App framework
st.title ('🦜🔗SEM Ad Copy Generator')
prompt = st.text_input('Plug in your prompt here')

#Prompt Templates 
headline_template = PromptTemplate(
        input_variables = ['topic'],
        template='Write an ad copy in less than 50 words based on {topic}'
)

ad_copy_template = PromptTemplate(
        input_variables = ['headline', 'wikipedia_research'],
        template='write an ad copy in less than 100 words based on this headline HEADLINE: {headline} while leveraging this wikipedia research:{wikipedia_research}'
)

# Memory
headline_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
ad_copy_memory = ConversationBufferMemory(input_key='headline', memory_key='chat_history')

# LLMs
llm = OpenAI(temperature=0.7)
headline_chain = LLMChain(llm=llm, prompt=headline_template, verbose=True, output_key='headline', memory = headline_memory)
ad_copy_chain = LLMChain(llm=llm, prompt=ad_copy_template, verbose=True, output_key='ad_copy', memory=ad_copy_memory)
#sequential_chain = SequentialChain(chains=[headline_chain, ad_copy_chain], input_variables=['topic'], output_variables=['headline', 'ad_copy'], verbose=True)

wiki = WikipediaAPIWrapper()

#show stuff if there is a prompt
if prompt:
        #response = sequential_chain({'topic':prompt})
        headline = headline_chain.run(prompt)
        wiki_research = wiki.run(prompt)
        ad_copy = ad_copy_chain.run(headline=headline, wikipedia_research=wiki_research)
        
        st.write(headline)
        st.write(ad_copy)

        with st.expander('Headline History'):
                st.info(headline_memory.buffer)

        with st.expander('Ad Copy History'):
                st.info(ad_copy_memory.buffer)

        with st.expander('Wikipedia Research History'):
                st.info(wiki_research)
        
