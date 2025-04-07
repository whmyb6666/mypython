from langchain.llms import Ollama
ollama = Ollama(base_url='http://localhost:11434',model="llama3-Chinese:8B")
print(ollama("你好"))
print(ollama("请翻译中文:hello world!"))
print(ollama("蓝牙耳机坏了，去医院挂牙科还是耳科？"))