from langchain_openai_like import init_openai_like_chat_model


system = "你是一个智能助手，能够帮助用户完成各种任务。"
messages = [("system", system)]

model = init_openai_like_chat_model(model="glm-4.5", provider="zai")

response = model.invoke(messages)
print(response.content)
