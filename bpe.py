from importlib.metadata import version
import tiktoken

print("version of tiktoken", version("tiktoken"))

tokenizer = tiktoken.get_encoding("cl100k_base")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
)

integers = tokenizer.encode(
    text,
    allowed_special={"<|endoftext|>"}
)

print(integers)