# local-rag-assistant

local-rag-assistant

## Project Setup

* Make an environment with python=3.10 using the following command 
``` bash
python -m venv venv
```
* Activate the environment
``` bash
venv\Scripts\activate
``` 

* Install the project dependencies using the following command 
```bash
pip install -r requirements.txt
```
* Run ingest.py with the sample Pdf file to generate Vectors and save to DB
``` python
python ingest.py
```
* Run the chat.py file To chat with your llm
```python
python chat.py
```
Note: do not forget to pull  models form ollama model names are included in models.txt

## License

**Copyright © 2025 DevOptima (Mustafa Dadikhi)**
All rights reserved. Unauthorized use, reproduction, or distribution of this material is prohibited without prior written permission from the copyright holder.
