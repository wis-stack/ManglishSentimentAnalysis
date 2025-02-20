from preprocessing import Preprocessing

def simple_tokenize(text):
    return text.split()

preprocessor = Preprocessing(simple_tokenize)

test_text = "Hey there! 😊 我不会去100 places, but I’ll visit 3. Email me at test@example.com."

processed_tokens = preprocessor.preprocessing_pipeline(test_text)
print(processed_tokens)
