import csv
import warnings
import io
import pathlib
from typing import Union
import os
import random
from PIL import Image
# import whisper
import openai
import gradio as gr
from transformers import pipeline
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from pytube import YouTube
from pytube import Search
from serpapi import GoogleSearch
import grpc
from langchain.embeddings.openai import OpenAIEmbeddings
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
import docx



openai.api_key  = os.environ['OPENAI_API_KEY']
stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'], #os.environ("STABILITY_KEY"), # key=os.environ['STABILITY_KEY'], # API Key reference.
    verbose=True, # Print debug messages.
    engine="stable-diffusion-v1-5", # Set the engine to use for generation.
    # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0
    # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-inpainting-v1-0 stable-inpainting-512-v2-0
)


whisper_from_pipeline = pipeline("automatic-speech-recognition",model="openai/whisper-medium")
EMBEDIDNGS = None
DATAFRAME_FILE = None
DOCSEARCH = None
RANDOM_USER = ''.join(chr(random.randint(65,90)) for i in range(8))+f'{random.randint(1,10000000000)}'
print(f'{RANDOM_USER} chat started')

############# FUNCTION DEPENDING ON IPYTHON FUNCTIONS FROM OPENAI RESPONSE
def gen_draw(user_query:str)->tuple:
  ###USES STABLE DIFFUSION
  answers = stability_api.generate(
                        prompt = user_query,
                        seed=992446758, # If a seed is provided, the resulting generated image will be deterministic.
                                # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
                                # Note: This isn't quite the case for Clip Guided generations, which we'll tackle in a future example notebook.
                        steps=30, # Amount of inference steps performed on image generation. Defaults to 30.
                        cfg_scale=8.0, # Influences how strongly your generation is guided to match your prompt.
                            # Setting this value higher increases the strength in which it tries to match your prompt.
                            # Defaults to 7.0 if not specified.
                        width=512, # Generation width, defaults to 512 if not included.
                        height=512, # Generation height, defaults to 512 if not included.
                        samples=1, # Number of images to generate, defaults to 1 if not included.
                        sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
                                                            # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
                                                            # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m)
                        )
  try:
    for resp in answers:
      for artifact in resp.artifacts:
        if artifact.finish_reason == generation.FILTER:
          warnings.warn(
              "Your request activated the API's safety filters and could not be processed."
              "Please modify the prompt and try again.")
        if artifact.type == generation.ARTIFACT_IMAGE:
          img = Image.open(io.BytesIO(artifact.binary))
          image_file = f'/tmp/{artifact.seed}.png'
          img.save(image_file)
          return (image_file,)
  except grpc._channel._MultiThreadedRendezvous as e:
    print(f'Exception : {e.__class__}')
    print(e)
    return "Invalid prompt"


def vid_tube(user_query:str) -> tuple:
  py_tube_list_of_videos = Search(user_query)
  first_video = py_tube_list_of_videos.results[0]
  yt_flag = False
  for vid in py_tube_list_of_videos.results:
    print(vid.vid_info.keys())
    if vid.vid_info.get('streamingData'):
      print(vid.vid_info.keys(),'-')
      yt_flag = True
      file_path = vid.streams.get_highest_resolution().download('/tmp/')
      break
  
  return (file_path,) if yt_flag else "The system cannot fulfill your request currently please try later"


def search_internet(user_query:str,*,key_number:int) -> str:
  if key_number >= 9:
    raise gr.Error("Out of Google API Keys")
  try:
    params = {
            "q": user_query,
            "location": "Bengaluru, Karnataka, India",
            "hl": "hi",
            "gl": "in",
            "google_domain": "google.co.in",
            # "api_key": ""
            "api_key": os.environ[f'GOOGLE_API{key_number}'] #os.environ("GOOGLE_API") #os.environ['GOOGLE_API']
        }
    search = GoogleSearch(params)
    results = search.get_dict()
    print(results)
    organic_results = results["organic_results"]
    print(f"Key {key_number} used")
        
    
    snippets = ""
    counter = 1
    for item in organic_results:
      snippets += str(counter) + ". " + item.get("snippet", "") + '\n' + item['about_this_result']['source']['source_info_link'] + '\n'
      counter += 1
    
        # snippets
    
    response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f'''following are snippets from google search with these as knowledge base only answer questions and print  reference link as well followed by answer. \n\n {snippets}\n\n question-{user_query}\n\nAnswer-''',
            temperature=0.49,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)

        
    result = response.choices[0].text
    
  except Exception as e:
    print(f'search google: ')
    print(f'GOOGLE_API{key_number} OUT OF LIMIT!')
    print(f'Exception: {e.__class__}, {e}')
    return search_internet(user_query,key_number = key_number+1)
  return result

def search_document_uploaded(user_query:str) -> str:
   print('Searching uploaded document......')
   # docsearch = FAISS.load_local(folder_path = f'/tmp/{RANDOM_USER}embeddings',embeddings=EMBEDIDNGS)
   chain = load_qa_chain(OpenAI(), chain_type="stuff")
   docs = DOCSEARCH.similarity_search(user_query)
   return chain.run(input_documents=docs, question=user_query)


def ask_dataframes(user_query):
  return DATAFRAME_FILE.run(user_query)

############# GET OPENAI RESPONSE
def get_open_ai_reponse(user_query:str)->Union[tuple,str]:
  print(EMBEDIDNGS)
  if (EMBEDIDNGS is not None) and (DOCSEARCH is not None):
    print('Searching document')
    return search_document_uploaded(user_query)
  
  if DATAFRAME_FILE is not None:
    print('Dataframe')
    return ask_dataframes(user_query)

    
  open_ai_response = openai.Completion.create(
                model="text-davinci-003",
                prompt=f'''Your name is GenZBot  and  knowledge cutoff date is 2021-09, and you are not aware of any events after that time. if the  
                    Answer to following questions is not from your knowledge base or in case of queries like date, time, weather 
                      updates / stock updates / current affairs / news or people which requires you to have internet connection  then print i don't have access to internet to answer your question, 
                      if  question is related to  image or  painting or drawing or diagram generation then print ipython type output function gen_draw("detailed prompt of image to be generated")
                      if the question is related to playing a song or video or music of a singer then print ipython type output  function vid_tube("relevent search query")
                      if the question is related to operating home appliances then print ipython type output function home_app(" action(ON/Off),appliance(TV,Geaser,Fridge,Lights,fans,AC)") . 
                      if question is realted to sending mail or sms then print ipython type output function messenger_app(" message of us ,messenger(email,sms)")
                      \nQuestion-{user_query}
                      \nAnswer -''',
                temperature=0.49,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
                )
  result_from_open_ai = open_ai_response.choices[0].text
  if 'gen_draw' in result_from_open_ai:
    result = gen_draw(user_query) ## will write drawn image to file
    
  elif 'vid_tube' in result_from_open_ai:
    try:
      result = vid_tube(user_query) ## play youtube video
    except KeyError as e:
      print(e)
      result = "The system is spacing an issue please try again later"
  
  elif ("don't" in result_from_open_ai)  or ("internet" in result_from_open_ai):
    result = search_internet(user_query,key_number = 1) 
  else:
    result = result_from_open_ai
  return result


############### DIFFERENT OUTPUT FUNCTIONS
def user_input(chat_history:list,user_query:str)->list:
  result =  get_open_ai_reponse(user_query)
  print(f'user_input: {chat_history + [(user_query,result)]}')  
  return chat_history + [(user_query,result)]

def transcribe(chat_history:list,user_audio_query:str)->list:
  print(user_audio_query.__class__)
  # text_from_speech = p(user_audio_query)["text"]
  try:
    user_query_from_audio = whisper_from_pipeline(user_audio_query)["text"]
  except Exception as e:
    print('EXCEPTION AS E')
    result = f'We are having a problem : {e}'
  else:
    result = get_open_ai_reponse(user_query_from_audio)
    
    # user_query_from_audio if user_query_from_audio else result
  print(result)
  print(f'transcribe: {chat_history + [(user_query_from_audio,result)]}')  
  return chat_history + [(user_query_from_audio,result)]


def pdf(file_name):
  print(f'Processing {file_name} pdf file')
  reader = PdfReader(file_name)
  raw_text = ''
  for i, page in enumerate(reader.pages):
      text = page.extract_text()
      if text:
          raw_text += text
  text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
  )
  texts = text_splitter.split_text(raw_text)
  return texts

def docx_file(file_name):
  print(f'Processing .docx file: {file_name}')
  doc = docx.Document(file_name)

  # iterate over paragraphs and print their text
  raw_text = ''
  for para in doc.paragraphs:
      raw_text += para.text
  text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
  )
  texts = text_splitter.split_text(raw_text)
  return texts

def text_file(file_name):
  print('Processing text file')
  with open(file_name) as file:
    raw_text = ''
    for line in file:
      raw_text += line
  text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
  )
  texts = text_splitter.split_text(raw_text)
  return texts
  

  


def build_embeddings(file_name,file_ext):
  
  
  functions_by_file_type = { 'pdf': pdf,
            'docx': docx_file,
           'txt':  text_file
      
  }
  
  texts =  functions_by_file_type.get(file_ext.replace('.','').strip())(file_name)
  print(texts)

  global EMBEDIDNGS 
  EMBEDIDNGS = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
  global DOCSEARCH  
  DOCSEARCH = FAISS.from_texts(texts, EMBEDIDNGS)
  # if not os.path.exists(f'/tmp/{RANDOM_USER}embeddings'):
  #   os.mkdir(f'/tmp/{RANDOM_USER}embeddings')
  # docsearch.save_local(f'/tmp/{RANDOM_USER}embeddings')
  # print(f'Embeddings created to /tmp/{RANDOM_USER}embeddings')
  
 
def ask_questions_abt_dataframes(file,file_ext):
  print(file_ext)
  global EMBEDIDNGS
  EMBEDIDNGS = None
  global DOCSEARCH
  DOCSEARCH = None

  

  reader_function = { '.csv': pd.read_csv, '.xlsx': pd.read_excel }.get(file_ext)
  print(reader_function.__name__)
  global DATAFRAME_FILE
  DATAFRAME_FILE = create_pandas_dataframe_agent(
    OpenAI(openai_api_key=os.environ['OPENAI_API_KEY']),
    reader_function(file.name)
  )




def upload_file(chatbot_history,file_uploaded):
  file_ext = os.path.splitext(file_uploaded.name)[-1]
  if file_ext not in ['.csv','.docx','.xlsx','.pdf','.txt']:
    return  chatbot_history + [(None, 'Invalid file format. We currently only csv, docx, pdf, txt, xlsx file extensions.')]  

  print(file_uploaded.__class__)
  
  if file_ext not in ['.csv','.xlsx']:
    build_embeddings(file_uploaded.name,file_ext)
  else:
    try:
        ask_questions_abt_dataframes(file_uploaded,file_ext)
    except Exception as e:
        print(f'Dataframes {e}')
        return  chatbot_history + [(None, f'Kindly attempt again at a subsequent time.')]
  
  
  return chatbot_history + [(None, f'You have uploaded {os.path.split(file_uploaded.name)[-1]} successfully. You can start asking questions about the document.If you want to stop asking questions about the uploaded document click on "clear chat history".')]


def clear_chat_history(history:list)->list:
  history.clear()
  global EMBEDIDNGS
  EMBEDIDNGS = None

  global DATAFRAME_FILE 
  DATAFRAME_FILE = None
    
  global DOCSEARCH
  DOCSEARCH = None
    
  # storing_folder = pathlib.Path('/tmp/')
  # for file in storing_folder.iterdir():
  #   if file.is_file():
  #     print(f'{file} to be deleted')  
  #     file.unlink()
  #     print(f'{file} deleted')   

  # global EMBEDIDNGS
  # EMBEDIDNGS = None

  # global DATAFRAME_FILE 
  # DATAFRAME_FILE = None
  return history





#################### DRIVER SCRIPT #####################
with gr.Blocks(theme='freddyaboulton/test-blue') as demo:
   
  gr.Markdown("""<h1 style="color:skyblue;font-family:'Brush Script MT', cursive;text-align:center">GenZBot</h1>""")
  gr.Markdown("""GenZBot is a virtual assistant that employs advanced artificial intelligence (AI) technologies to enhance its capabilities. Utilizing cutting-edge AI techniques such as Whisper, chatgpt, internet, Dall-E and OpenAI and Langchain, GenZBot can provide users with a wide range of useful features. By leveraging AI, GenZBot can understand and respond to users' requests in a natural and intuitive manner, allowing for a more seamless and personalized experience. Its ability to generate paintings, drawings, and abstract art, play music and videos, and you can Upload your documents and ask questions about the document, is made possible by sophisticated AI algorithms that can produce complex and nuanced results. Overall, GenZBot's extensive use of AI technology enables it to serve as a powerful and versatile digital assistant that can adapt to the needs of its users.""")
  chatbot = gr.Chatbot()
  
  with gr.Row():
    with gr.Column():
      user_text_query = gr.Text(label="Your Query",placeholder="Your Query")
    with gr.Column(scale=0.15, min_width=0):#
      user_audio_microphone_query = gr.Audio(label="Record",source="microphone",type="filepath")
      user_audio_microphone_submit_button = gr.Button("Get me result")
    with gr.Column(scale=0.15, min_width=0):
      upload_button = gr.UploadButton("📁", info="Upload text files and start talking to them")
      gr.Markdown("Upload document by clicking on the directory icon.")  
  clear_button = gr.Button("Clear chat history")
    

  user_text_query.submit(fn=user_input,inputs=[chatbot,user_text_query],outputs=[chatbot])
  user_audio_microphone_submit_button.click(fn=transcribe,inputs=[chatbot,user_audio_microphone_query],outputs=[chatbot])
  clear_button.click(fn=clear_chat_history,inputs=[chatbot],outputs=[chatbot])
  upload_button.upload(upload_file,inputs=[chatbot,upload_button],outputs=[chatbot])

    
 


demo.launch(debug=True)
