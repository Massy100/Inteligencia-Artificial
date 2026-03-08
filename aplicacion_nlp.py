import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset  
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
from tqdm import tqdm

# Configuracion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Usando dispositivo: {device}')

# 1. DATOS DE EJEMPLO (comentarios en español)
comentarios = [
    # Positivos (1)
    "Me encantó este producto, es excelente",
    "Muy buen servicio, lo recomiendo totalmente",
    "La atención al cliente fue increíble",
    "Excelente calidad, superó mis expectativas",
    "Me siento muy satisfecho con mi compra",
    "El producto llegó antes de lo esperado",
    "Totalmente recomendado, funciona perfecto",
    "La mejor compra que he hecho este año",
    "El personal es muy amable y profesional",
    "Volvería a comprar sin dudarlo",
    
    # Negativos (0)
    "Pésimo servicio, no lo recomiendo",
    "El producto llegó roto y en mal estado",
    "Muy mala experiencia, no volveré a comprar",
    "La atención al cliente es terrible",
    "No funciona como esperaba, decepcionado",
    "El envío tardó mucho más de lo debido",
    "Producto de baja calidad, no vale lo que cuesta",
    "Me arrepiento de haber comprado esto",
    "El servicio al cliente no responde",
    "No cumple con lo prometido, muy malo"
]

etiqueta = [1] * 10 + [0] * 10  # 1 para positivo, 0 para negativo
print(etiqueta)

# Crea un DataFrame
df = pd.DataFrame({'comentario': comentarios, 'sentimiento': etiqueta})
print(df)

# DataSet personalizado para python
class SentimentDataset(Dataset):
    def __init__(self, textos, etiquetas, tokenizer, max_len=128):
        self.textos = textos
        self.etiquetas = etiquetas
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, idx):
        texto = self.textos[idx]
        etiqueta = self.etiquetas[idx]
        
        # Tokenizar el texto
        encoding = self.tokenizer(
            texto,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_id': encoding['input_id'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(etiqueta, dtype=torch.long)
        }
        
# 3. Preparar los datos
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
# 4. Dividir los datos en entrenamiento y prueba
x = np.array(comentarios)
y = np.array(etiqueta, dtype=np.int64)
X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3, 
    random_state=23,
    stratify=y,
)
    
print(f'Tamano del conjunto de entrenamiento: {len(X_train)}')
print(f'Tamano del conjunto de prueba: {len(X_test)}')
    
train_dataset = SentimentDataset(X_train, y_train, tokenizer)
test_dataset = SentimentDataset(X_test, y_test, tokenizer)
print(test_dataset)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)

# Entrenamiento del modelo
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', 
    num_labels=2,
    )
model = model.to(device)

optimizador = AdamW(model.parameters(), lr=2e-5)
# Funcion de entrenamiento
def entrenar_modelo(model, train_loader, test_loader, optimizer, epocas=5):
    train_losses = []
    test_accuracies = []
    for epoca in range(epocas):
        print(f'Epoca {epoca + 1}/{epocas}:')
        print('=' * 50)
        # Modo entrenamiento
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate (tqdm(train_loader, desc='Entrenando')):
            # Mover los datos al dispositivo
            input_id = batch['input_id'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            ## Pasar 
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_id, 
                attention_mask=attention_mask, 
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss
            num_batches += 1
            
            ## Pasar hacia atrás y optimizar
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_loss.item())
        
        # Evaluacion
        accuracy = evaluar_modelo(model, test_loader)
        test_accuracies.append(accuracy)
        print(f'Perdida promedio: {avg_loss:.4f} | Precision en prueba: {accuracy:.4f}')

    return train_losses, test_accuracies

def evaluar_modelo(model, test_loader):
    model.eval()
    predicciones = []
    verdaderos = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluando")):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                _, preds = torch.max(outputs.logits, dim=1)
                predicciones.extend(preds.cpu().tolist())
                verdaderos.extend(labels.cpu().tolist())
            except:
                print(f'Error evaluando {batch_idx}: {e}')
                continue
    
    accuracy = accuracy_score(verdaderos, predicciones)
    return accuracy
