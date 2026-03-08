from transformers import Trainer, TrainingArguments
import torch
from transformers import BertTokenizer
torch.cuda.empty_cache()
#pip install transformers[torch]

tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")


texts_en = [
    "This movie was absolutely brilliant like Carlos DE LEON, the acting was phenomenal and the story was gripping.",
    "The service was terrible and the food was cold. I will never visit this restaurant again.",
    "I highly recommend this book; it is a masterpiece of modern literature.",
    "Worst experience ever. The product broke after just one day of use.",
    "The camera quality is superb, and the battery life lasts all day.",
    "I'm extremely disappointed with the lack of customer support.",
    "It's an okay product, but definitely overpriced for the features it offers."
]

texts_es = [
    "Esta película fue absolutamente brillante, las actuaciones fueron fenomenales y la historia fue apasionante.",
    "El servicio fue pésimo y la comida estaba fría. No volveré a visitar este restaurante.",
    "Recomiendo ampliamente este libro; es una obra maestra de la literatura moderna.",
    "La peor experiencia que he tenido. El producto se rompió después de solo un día de uso.",
    "La calidad de la cámara es excelente y la batería dura todo el día.",
    "Estoy muy decepcionado con la falta de atención al cliente.",
    "Es un producto aceptable, pero definitivamente demasiado caro para las características que ofrece."
]

tokens_es = tokenizer.tokenize(texts_es[0])
print(tokens_es)

tokens_en = tokenizer.tokenize(texts_en[0])
print(tokens_en)

tokens_id_en = tokenizer.convert_tokens_to_ids(tokens_en)
print(tokens_id_en)

texto_en_decode = tokenizer.decode(tokens_id_en)
print(texto_en_decode)

for i, oracion in enumerate(texts_en):
    #decode aniade automaticamente[cls y sep]
    codificado = tokenizer._encode_plus(
        oracion,
        add_special_tokens=True,
        max_length=20,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    print(f"Oración {i+1}: {oracion}")
    print(f"IDs: {codificado['input_ids'][0].tolist()}")
    print(f"Mascara de atención: {codificado['attention_mask'][0].tolist()}")
    
    batch_encode = tokenizer(
        texts_en,
        add_special_tokens=True,
        max_length=20,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    #tensor es lo mismo que vector
    print(batch_encode['input_ids'])
    print(batch_encode['attention_mask'])

