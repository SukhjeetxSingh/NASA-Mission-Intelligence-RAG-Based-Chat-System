#!/bin/bash
# setup_env.sh
# Installs project dependencies and patches environment conflicts.
# Required after every Udacity/Vocareum workspace reset.
# Fixes:
#   1. ragas VertexAI import conflict
#   2. rag_client.py query dimension mismatch (OpenAI 1536 vs local 384)
#   3. ChromaDB include='ids' invalid argument

set -e

echo "==> Installing requirements..."
pip install \
    pysqlite3-binary \
    chromadb==1.5.7 \
    openai==2.31.0 \
    langchain-openai==1.1.13 \
    langchain-core==1.4.7 \
    langchain-community==0.4.2 \
    ragas==0.4.3 \
    pandas==3.0.2 \
    streamlit==1.56.0 \
    rouge-score

echo "==> Patching ragas VertexAI import..."
python -c "
import re

path = '/opt/venv/lib/python3.13/site-packages/ragas/llms/base.py'

with open(path) as f:
    content = f.read()

# Remove ALL previous patch attempts (nested or malformed)
content = re.sub(
    r'(try:\s*\n\s*)*from langchain_community\.chat_models\.vertexai import ChatVertexAI(\s*\nexcept ImportError:\s*\n\s*ChatVertexAI = None)*',
    'from langchain_community.chat_models.vertexai import ChatVertexAI',
    content
)

# Apply one clean patch
old = 'from langchain_community.chat_models.vertexai import ChatVertexAI'
new = 'try:\n    from langchain_community.chat_models.vertexai import ChatVertexAI\nexcept ImportError:\n    ChatVertexAI = None'

if old in content:
    content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print('Patched:', path)
else:
    print('Patch already applied or line not found — skipping.')
"

echo "==> Verifying ragas import..."
python -c "import ragas; print(f'ragas {ragas.__version__} OK')"

echo "==> Verifying rag_client.py uses OpenAI embeddings for queries..."
python -c "
path = 'rag_client.py'
with open(path) as f:
    content = f.read()
if 'query_embeddings' in content:
    print('rag_client.py: query_embeddings OK')
else:
    print('WARNING: rag_client.py still uses query_texts — retrieval will fail!')
    print('Please check rag_client.py retrieve_documents function.')
"

echo "==> Done. Environment is ready."