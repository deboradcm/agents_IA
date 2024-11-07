import os

# Be sure to have all the API keys in your local environment as shown below
# Do not publish environment keys in production
# os.environ["OPENAI_API_KEY"] = "sk"
# os.environ["FIREWORKS_API_KEY"] = ""
# os.environ["MONGO_URI"] = ""
FIREWORKS_API_KEY = os.environ.get("fw_3ZQw2kWUAhALsSFhuYoweB7x")
QROQ_LLHAMA_API_KEY = os.environ.get("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")
MONGO_URI = os.environ.get("MONGO_URI")