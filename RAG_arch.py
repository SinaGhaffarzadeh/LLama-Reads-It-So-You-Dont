
'''
This code implement by llamaindex and sentence-transforemr libraries.
'''

# Importing libraries
import torch
from huggingface_hub import whoami, login
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from llama_index.core.node_parser import SimpleNodeParser 


# Checking availibity of Cuda on system
print('Cuda is available!', torch.cuda.is_available())  # Should return True
print("The version of Cuda is:",torch.version.cuda)         # Should match something like '12.1'

# # Downloading the model using its ID, which is taken from Huggingface.
# model_id = "Orenguteng/Llama-3-8B-Lexi-Uncensored" # Main model
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)

# # Downloading embedding model using "SentenceTransformer" library (models: intfloat/e5-small-v2  , e5-base-v2 )
# emb_model = SentenceTransformer('all-MiniLM-L6-v2') 
# print("Model downloaded and cached.")

# Loading pdf files from directory
documents = SimpleDirectoryReader("Data").load_data() 

# Parsing and Indexing all data
parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents(documents)

# Summoning Embedding model and Embedding all indexed data 
embed_model = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2")
index = VectorStoreIndex(nodes, embed_model=embed_model)

# LLM 

'''
Decoder-based models like llama, microsoft, etc. have less limitation in token. Therefore, they will have good performance in generating output. on the other hand, Encoder-
Decoder based models like T5 or BERT have limitation that means they will not have good perfomance because they are not able to read much paragraph.
'''
try:
    llm = HuggingFaceLLM(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", # mistralai/Mistral-7B-Instruct-v0.3 , TinyLlama/TinyLlama-1.1B-Chat-v1.0
        tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device_map="cuda",  # or "cuda" if using GPU
        max_new_tokens=128,
        context_window=1024,  # or even lower
        generate_kwargs={"temperature": 0.3, "do_sample": True}, # low tempeture make model to choice the answer with highest possibility.
                                                                # On the other hand the highest tempeture make diversity in choosing.
                                                                # for example "My favorite food is ..." the model by low temperture will say
                                                                # pizza becuase it has highest possibility but if the temperture setted on highest 
                                                                # it will pick the answer from other foods like tacos.
                                                                # Moreover by running code every time we will get to different answer. 
                                                                # So for having reliable and predictable model setting 0 for temperture will be suitable
                                                                # otherwise for the tasks that require verity setting a value between 0 to 1 will be suitable.
    )

    # Logining to API
    login(token="....", add_to_git_credential=False) # Add_your_HuggingFace_Token
    print(whoami())

    # Query system baised on Language model
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query("what is motion?")
    print(response)
    
except Exception as e:
    print("Error loading LLM or tokenizer:", e)
    exit()
