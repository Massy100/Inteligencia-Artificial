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
            'input_ids': encoding['input_ids'].flatten(),
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
            input_id = batch['input_ids'].to(device)
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

# Entrenar el modelo
train_losses, test_accuracies = entrenar_modelo(
    model, train_dataloader, test_dataloader, optimizador, epocas=10
)

# Evaluación final detallada
model.eval()
all_predicciones = []
all_verdaderos = []
all_probabilidades = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Obtener probabilidades con softmax
        probabilidades = torch.nn.functional.softmax(outputs.logits, dim=1)
        _, preds = torch.max(outputs.logits, dim=1)
        
        all_predicciones.extend(preds.cpu().tolist())
        all_verdaderos.extend(labels.tolist())
        all_probabilidades.extend(probabilidades.cpu().tolist())

# Reporte de clasificación
print("\n" + "="*50)
print("REPORTE DE CLASIFICACIÓN")
print("="*50)
print(classification_report(
    all_verdaderos, 
    all_predicciones,
    target_names=['Negativo', 'Positivo']
))

# Mostrar algunos ejemplos con predicciones
print("\n" + "="*50)
print("EJEMPLOS DE PREDICCIONES")
print("="*50)
indices_prueba = np.random.choice(len(X_test), size=5, replace=False)
for idx in indices_prueba:
    comentario = X_test[idx]
    real = "POSITIVO" if y_test[idx] == 1 else "NEGATIVO"
    pred = "POSITIVO" if all_predicciones[idx] == 1 else "NEGATIVO"
    prob_pos = all_probabilidades[idx][1]
    prob_neg = all_probabilidades[idx][0]
    
    print(f"\nComentario: {comentario}")
    print(f"Real: {real}")
    print(f"Predicción: {pred}")
    print(f"Probabilidad Positivo: {prob_pos:.4f}")
    print(f"Probabilidad Negativo: {prob_neg:.4f}")
    print(f"{'_/' if real == pred else 'X'} Correcto: {real == pred}")
    
def predecir_sentimiento(texto, model, tokenizer, device):
    """
    Predice el sentimiento de un nuevo comentario
    """
    model.eval()
    
    # Tokenizar
    encoding = tokenizer(
        texto,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Mover a dispositivo
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predecir
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        probabilidades = torch.nn.functional.softmax(outputs.logits, dim=1)
        prediccion = torch.argmax(probabilidades, dim=1)
        
        prob_pos = probabilidades[0][1].item()
        prob_neg = probabilidades[0][0].item()
    
    sentimiento = "POSITIVO" if prediccion.item() == 1 else "NEGATIVO"
    confianza = max(prob_pos, prob_neg)
    
    return {
        'sentimiento': sentimiento,
        'confianza': confianza,
        'probabilidad_positivo': prob_pos,
        'probabilidad_negativo': prob_neg
    }
# Listado de comentarios nuevos
# Probar con nuevos comentarios
print("\n" + "="*50)
print("PREDICCIONES EN TIEMPO REAL")
print("="*50)
nuevos_comentarios = [
    "Este producto es increíble, me encantó",
    "Una verdadera porquería, no funciona",
    "Más o menos, podría ser mejor",
    "Excelente atención, muy recomendable",
    "No me gustó para nada, muy decepcionante",
    "Recomendable"
]

for comentario in nuevos_comentarios:
    resultado = predecir_sentimiento(comentario, model, tokenizer, device)
    print(f"\nComentario: {comentario}")
    print(f"Sentimiento: {resultado['sentimiento']}")
    print(f"Confianza: {resultado['confianza']:.2%}")
    print(f"Positivo: {resultado['probabilidad_positivo']:.2%}, "
          f"Negativo: {resultado['probabilidad_negativo']:.2%}")