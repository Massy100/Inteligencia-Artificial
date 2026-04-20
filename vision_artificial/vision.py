import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import pytesseract
import os
import numpy as np
import re
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ==================== VALIDACIÓN DE PLACAS GUATEMALA ====================

TIPOS_VEHICULOS = {
    'P': 'Particular',
    'M': 'Motocicleta',
    'A': 'Alquiler',
    'C': 'Comercial',
    'U': 'Urbano/Oficial',
    'T': 'Remolque'
}

PATRON_PLACA_GT = re.compile(r'^[A-Z]\d{3}[A-Z]{3}$')

def corregir_placa(texto):
    """Corrige errores comunes en placas guatemaltecas"""
    if not texto or len(texto) != 7:
        return texto, False
    
    texto = texto.upper().strip()
    texto_corregido = list(texto)
    
    for i, char in enumerate(texto_corregido):
        if i == 0:  # Letra inicial
            if char == '1':
                texto_corregido[i] = 'P'
            elif char == '0':
                texto_corregido[i] = 'O'
            elif char == '2':
                texto_corregido[i] = 'Z'
            elif char == '3':
                texto_corregido[i] = 'E'
            elif char == '4':
                texto_corregido[i] = 'A'
            elif char == '5':
                texto_corregido[i] = 'S'
            elif char == '6':
                texto_corregido[i] = 'G'
            elif char == '8':
                texto_corregido[i] = 'B'
        elif i in [1, 2, 3]:  # Números
            if char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                if char == 'O':
                    texto_corregido[i] = '0'
                elif char == 'I':
                    texto_corregido[i] = '1'
                elif char == 'Z':
                    texto_corregido[i] = '2'
                elif char == 'E':
                    texto_corregido[i] = '3'
                elif char == 'A':
                    texto_corregido[i] = '4'
                elif char == 'S':
                    texto_corregido[i] = '5'
                elif char == 'G':
                    texto_corregido[i] = '6'
                elif char == 'B':
                    texto_corregido[i] = '8'
        else:  # Letras finales
            if char in '0123456789':
                if char == '0':
                    texto_corregido[i] = 'O'
                elif char == '1':
                    texto_corregido[i] = 'I'
                elif char == '2':
                    texto_corregido[i] = 'Z'
                elif char == '3':
                    texto_corregido[i] = 'E'
                elif char == '4':
                    texto_corregido[i] = 'A'
                elif char == '5':
                    texto_corregido[i] = 'S'
                elif char == '6':
                    texto_corregido[i] = 'G'
                elif char == '8':
                    texto_corregido[i] = 'B'
    
    texto_final = ''.join(texto_corregido)
    return texto_final, texto_final != texto

def validar_placa_guatemala(texto):
    """Valida formato de placa Guatemala"""
    if not texto:
        return False, None, "No detectado", texto
    
    if PATRON_PLACA_GT.match(texto):
        letra = texto[0]
        tipo = TIPOS_VEHICULOS.get(letra, 'Tipo desconocido')
        return True, tipo, f"Válida - {letra}: {tipo}", texto
    
    texto_corr, corregido = corregir_placa(texto)
    if PATRON_PLACA_GT.match(texto_corr):
        letra = texto_corr[0]
        tipo = TIPOS_VEHICULOS.get(letra, 'Tipo desconocido')
        msg = f"Válida (corregida) - {letra}: {tipo}" if corregido else f"Válida - {letra}: {tipo}"
        return True, tipo, msg, texto_corr
    
    return False, None, "Formato inválido (debe ser L 999 LLL)", texto

def formatear_placa(texto):
    if len(texto) >= 7:
        return f"{texto[0]} {texto[1:4]} {texto[4:7]}"
    return texto

# ==================== CLASE PRINCIPAL ====================

class PlateReaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lector de Placas Vehiculares - Guatemala")
        self.root.geometry("1400x850")
        self.root.configure(bg='#2c3e50')
        
        self.images_paths = []
        self.current_image_index = 0
        self.processed_plates = []
        
        self.setup_styles()
        self.create_widgets()
        
    def setup_styles(self):
        self.bg_color = '#2c3e50'
        self.fg_color = '#ecf0f1'
        self.button_color = '#3498db'
        self.success_color = '#27ae60'
        self.error_color = '#e74c3c'
        self.warning_color = '#f39c12'
        self.info_color = '#1abc9c'
        
    def create_widgets(self):
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel izquierdo
        left_panel = tk.Frame(main_frame, bg=self.bg_color, width=250)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        tk.Label(left_panel, text="Imágenes Cargadas", font=("Arial", 14, "bold"), 
                bg=self.bg_color, fg=self.fg_color).pack(pady=10)
        
        self.images_listbox = tk.Listbox(left_panel, height=20, bg='#34495e', 
                                         fg=self.fg_color, selectmode=tk.SINGLE)
        self.images_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.images_listbox.bind('<<ListboxSelect>>', self.on_image_select)
        
        btn_frame = tk.Frame(left_panel, bg=self.bg_color)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Cargar Imágenes", command=self.load_images,
                 bg=self.button_color, fg='white', font=("Arial", 10, "bold")).pack(pady=5)
        tk.Button(btn_frame, text="Limpiar Lista", command=self.clear_images,
                 bg=self.error_color, fg='white', font=("Arial", 10, "bold")).pack(pady=5)
        
        # Panel central
        center_panel = tk.Frame(main_frame, bg=self.bg_color)
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        image_frame = tk.Frame(center_panel, bg='black', relief=tk.RAISED, bd=2)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.image_label = tk.Label(image_frame, bg='black')
        self.image_label.pack(expand=True)
        
        results_frame = tk.Frame(center_panel, bg=self.bg_color)
        results_frame.pack(fill=tk.X, pady=5)
        
        text_frame = tk.Frame(results_frame, bg=self.bg_color)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(text_frame, height=8, bg='#34495e', 
                                    fg=self.fg_color, font=("Courier", 10))
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(text_frame, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        
        control_frame = tk.Frame(center_panel, bg=self.bg_color)
        control_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(control_frame, text="◀ Anterior", command=self.prev_image,
                 bg=self.button_color, fg='white', font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Procesar Placa", command=self.process_current_image,
                 bg=self.success_color, fg='white', font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Siguiente ▶", command=self.next_image,
                 bg=self.button_color, fg='white', font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Exportar Resultados", command=self.export_results,
                 bg=self.warning_color, fg='white', font=("Arial", 10, "bold")).pack(side=tk.RIGHT, padx=5)
        
        # Panel derecho
        right_panel = tk.Frame(main_frame, bg=self.bg_color, width=450)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Sección de métodos
        tk.Label(right_panel, text="Métodos de Detección", font=("Arial", 14, "bold"),
                bg=self.bg_color, fg=self.fg_color).pack(pady=10)
        
        self.methods_frame = tk.Frame(right_panel, bg='#34495e', relief=tk.RAISED, bd=1)
        self.methods_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.method_labels = {}
        methods = ["Solo X", "Solo Y", "X+Y", "Color Blanco"]
        for method in methods:
            frame = tk.Frame(self.methods_frame, bg='#2c3e50', relief=tk.SUNKEN, bd=1)
            frame.pack(fill=tk.X, padx=5, pady=5)
            tk.Label(frame, text=f"{method}:", bg='#2c3e50', fg=self.fg_color,
                    font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
            label_result = tk.Label(frame, text="No procesado", bg='#2c3e50', 
                                   fg=self.warning_color, font=("Arial", 9))
            label_result.pack(side=tk.RIGHT, padx=5)
            self.method_labels[method] = label_result
        
        # Sección de validación
        tk.Label(right_panel, text="Validación de Placa", font=("Arial", 14, "bold"),
                bg=self.bg_color, fg=self.fg_color).pack(pady=(20, 10))
        
        validation_frame = tk.Frame(right_panel, bg='#34495e', relief=tk.RAISED, bd=1)
        validation_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.placa_label = tk.Label(validation_frame, text="Placa: --", bg='#34495e', 
                                    fg=self.info_color, font=("Arial", 11, "bold"))
        self.placa_label.pack(pady=5, padx=10, anchor=tk.W)
        
        self.tipo_label = tk.Label(validation_frame, text="Tipo: --", bg='#34495e', 
                                   fg=self.fg_color, font=("Arial", 10))
        self.tipo_label.pack(pady=2, padx=10, anchor=tk.W)
        
        self.estado_label = tk.Label(validation_frame, text="Estado: --", bg='#34495e', 
                                     fg=self.warning_color, font=("Arial", 10, "bold"))
        self.estado_label.pack(pady=5, padx=10, anchor=tk.W)
        
        # Info
        info_frame = tk.Frame(right_panel, bg='#2c3e50')
        info_frame.pack(fill=tk.X, pady=20)
        
        tk.Label(info_frame, text="Formato Guatemala", font=("Arial", 11, "bold"),
                bg='#2c3e50', fg=self.info_color).pack(pady=5)
        tk.Label(info_frame, text="Formato: L 999 LLL\nP: Particular | M: Motocicleta\nA: Alquiler | C: Comercial\nU: Urbano | T: Remolque",
                font=("Arial", 9), bg='#2c3e50', fg=self.fg_color, justify=tk.LEFT).pack(pady=5)
        
        # Barra de estado
        self.status_bar = tk.Label(self.root, text="Listo", bd=1, relief=tk.SUNKEN,
                                   anchor=tk.W, bg='#34495e', fg=self.fg_color)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    # ==================== MÉTODOS DE DETECCIÓN (DEL CÓDIGO QUE FUNCIONA) ====================
    
    def refinar_roi_placa(self, roi_gray):
        h, w = roi_gray.shape
        mejor_var = 0
        mejor_y = 0
        paso = max(1, h // 10)
        for yi in range(0, h - paso, paso):
            franja = roi_gray[yi:yi + paso, :]
            var = np.var(franja)
            if var > mejor_var:
                mejor_var = var
                mejor_y = yi
        y1 = max(0, mejor_y - paso)
        y2 = min(h, mejor_y + paso * 3)
        return roi_gray[y1:y2, :], y1

    def obtener_candidatos(self, gray, grad, w_img, h_img):
        grad = np.absolute(grad)
        grad = cv2.convertScaleAbs(grad)
        blur = cv2.GaussianBlur(grad, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:80]
        candidatos = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            ratio = w / float(h) if h != 0 else 0
            area = w * h
            if (1.8 < ratio < 6.5 and 1500 < area < w_img * h_img * 0.15
                    and w < w_img * 0.85 and y > h_img * 0.35):
                candidatos.append((x, y, w, h))
        return candidatos

    def candidatos_color_blanco(self, image_bgr, h_img):
        offset_y = int(h_img * 0.5)
        roi = image_bgr[offset_y:, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 40, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidatos = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            ratio = w / float(h) if h != 0 else 0
            area = w * h
            if 2.0 < ratio < 6.0 and area > 2000:
                candidatos.append((x, y + offset_y, w, h))
        return candidatos

    def ocr_roi(self, gray, x, y, w, h):
        roi = gray[y:y+h, x:x+w]
        ratio = w / float(h) if h != 0 else 0
        if ratio > 4.5 or (w * h) > 30000:
            roi, offset_y = self.refinar_roi_placa(roi)
            h = roi.shape[0]
        roi = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        roi = cv2.equalizeHist(roi)
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
        _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_text = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel_text)

        config_line = '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        config_word = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        texto = pytesseract.image_to_string(roi, config=config_line).strip().replace(" ", "").replace("\n", "")
        if len(texto) < 5:
            texto = pytesseract.image_to_string(roi, config=config_word).strip().replace(" ", "").replace("\n", "")
        return texto, roi

    def es_valido(self, texto):
        return 5 <= len(texto) <= 9

    def clave_orden(self, texto):
        distancia_a_7 = abs(len(texto) - 7)
        letras = sum(1 for c in texto if c.isalpha())
        return (distancia_a_7, -letras)

    def mejor_candidato(self, candidatos, gray, h_img):
        validos = []
        for (x, y, w, h) in candidatos:
            texto, roi = self.ocr_roi(gray, x, y, w, h)
            if not texto:
                continue
            if self.es_valido(texto):
                validos.append((texto, (x, y, w, h), roi))
        if not validos:
            return "", None, None
        validos.sort(key=lambda item: self.clave_orden(item[0]))
        mejor_texto, mejor_box, mejor_roi = validos[0]
        return mejor_texto, mejor_box, mejor_roi

    def process_plate(self, image_path):
        """Procesa la imagen usando gradientes y color blanco"""
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h_img, w_img = gray.shape

        # Gradientes
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_xy = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)

        gradientes = {"Solo X": grad_x, "Solo Y": grad_y, "X+Y": grad_xy}

        resultados = {}
        for nombre, grad in gradientes.items():
            candidatos = self.obtener_candidatos(gray, grad, w_img, h_img)
            texto, box, roi = self.mejor_candidato(candidatos, gray, h_img)
            resultados[nombre] = (texto, box, roi)

        # Color blanco
        cands_color = self.candidatos_color_blanco(image, h_img)
        texto, box, roi = self.mejor_candidato(cands_color, gray, h_img)
        resultados["Color Blanco"] = (texto, box, roi)

        # Elegir ganador global
        candidatos_finales = [
            (nombre, texto, box, roi)
            for nombre, (texto, box, roi) in resultados.items()
            if texto
        ]

        if candidatos_finales:
            candidatos_finales.sort(key=lambda item: self.clave_orden(item[1]))
            ganador, best_text, best_box, best_roi = candidatos_finales[0]
        else:
            ganador, best_text, best_box, best_roi = "—", "", None, None

        # Dibujar resultado en la imagen
        if best_box:
            x, y, w, h = best_box
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(image, best_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return {
            'imagen_procesada': image,
            'resultados': resultados,
            'ganador': ganador,
            'texto_detectado': best_text,
            'box': best_box,
            'roi': best_roi
        }
    
    # ==================== MÉTODOS DE INTERFAZ ====================
    
    def load_images(self):
        files = filedialog.askopenfilenames(
            title="Seleccionar imágenes",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if files:
            self.images_paths = list(files)
            self.current_image_index = 0
            self.update_images_list()
            self.display_current_image()
            self.update_status(f"Cargadas {len(self.images_paths)} imágenes")
    
    def update_images_list(self):
        self.images_listbox.delete(0, tk.END)
        for i, path in enumerate(self.images_paths):
            self.images_listbox.insert(tk.END, f"{i+1}. {os.path.basename(path)}")
    
    def on_image_select(self, event):
        selection = self.images_listbox.curselection()
        if selection:
            self.current_image_index = selection[0]
            self.display_current_image()
            self.clear_results()
    
    def display_current_image(self):
        if self.images_paths and 0 <= self.current_image_index < len(self.images_paths):
            img = cv2.imread(self.images_paths[self.current_image_index])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            h, w = img_rgb.shape[:2]
            max_h, max_w = 450, 650
            if h > max_h or w > max_w:
                scale = min(max_h/h, max_w/w)
                new_w, new_h = int(w*scale), int(h*scale)
                img_rgb = cv2.resize(img_rgb, (new_w, new_h))
            
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk
    
    def process_current_image(self):
        if not self.images_paths:
            messagebox.showwarning("Advertencia", "No hay imágenes cargadas")
            return
        
        image_path = self.images_paths[self.current_image_index]
        
        try:
            result = self.process_plate(image_path)
            
            if result and result['texto_detectado']:
                # Validar la placa detectada
                valido, tipo, mensaje, texto_corr = validar_placa_guatemala(result['texto_detectado'])
                placa_formateada = formatear_placa(texto_corr if valido else result['texto_detectado'])
                
                # Guardar resultado
                self.processed_plates.append({
                    'image': os.path.basename(image_path),
                    'method': result['ganador'],
                    'plate_original': result['texto_detectado'],
                    'plate_corrected': texto_corr if valido else result['texto_detectado'],
                    'plate_formatted': placa_formateada,
                    'is_valid': valido,
                    'vehicle_type': tipo,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Actualizar métodos
                for method, (text, box, roi) in result['resultados'].items():
                    if text:
                        self.method_labels[method].config(text=f"✓ {text}", fg=self.success_color)
                    else:
                        self.method_labels[method].config(text="No detectado", fg=self.error_color)
                
                # Actualizar validación
                self.placa_label.config(text=f"Placa: {placa_formateada}",
                                       fg=self.success_color if valido else self.warning_color)
                self.tipo_label.config(text=f"Tipo: {tipo if tipo else 'No válido'}")
                self.estado_label.config(text=f"Estado: {mensaje}",
                                        fg=self.success_color if valido else self.error_color)
                
                # Mostrar resultado
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END,
                    f"Imagen: {os.path.basename(image_path)}\n"
                    f"Mejor método: {result['ganador']}\n"
                    f"OCR detectado: {result['texto_detectado']}\n"
                    f"Placa corregida: {texto_corr if valido else result['texto_detectado']}\n"
                    f"Placa formateada: {placa_formateada}\n"
                    f"Válida: {'Sí' if valido else 'No'}\n"
                    f"Tipo: {tipo if tipo else 'N/A'}\n"
                    f"Procesado: {datetime.now().strftime('%H:%M:%S')}\n"
                    f"{'='*50}\n")
                self.results_text.see(tk.END)
                
                # Mostrar imagen procesada
                self.show_processed_image(result['imagen_procesada'])
                
                self.update_status(f"Placa: {placa_formateada}", 
                                  self.success_color if valido else self.warning_color)
            else:
                self.results_text.insert(tk.END, 
                    f"No se detectó placa en: {os.path.basename(image_path)}\n"
                    f"{'='*50}\n")
                for method in self.method_labels:
                    self.method_labels[method].config(text="No detectado", fg=self.error_color)
                self.placa_label.config(text="Placa: --")
                self.tipo_label.config(text="Tipo: --")
                self.estado_label.config(text="Estado: No detectada", fg=self.error_color)
                self.update_status("No se detectó placa", self.error_color)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar: {str(e)}")
            self.update_status(f"Error: {str(e)}", self.error_color)
    
    def show_processed_image(self, image):
        if image is not None:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            max_h, max_w = 450, 650
            if h > max_h or w > max_w:
                scale = min(max_h/h, max_w/w)
                new_w, new_h = int(w*scale), int(h*scale)
                img_rgb = cv2.resize(img_rgb, (new_w, new_h))
            
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk
    
    def clear_results(self):
        for method in self.method_labels:
            self.method_labels[method].config(text="No procesado", fg=self.warning_color)
        self.placa_label.config(text="Placa: --")
        self.tipo_label.config(text="Tipo: --")
        self.estado_label.config(text="Estado: --", fg=self.warning_color)
        self.results_text.delete(1.0, tk.END)
    
    def prev_image(self):
        if self.images_paths:
            self.current_image_index = (self.current_image_index - 1) % len(self.images_paths)
            self.display_current_image()
            self.clear_results()
    
    def next_image(self):
        if self.images_paths:
            self.current_image_index = (self.current_image_index + 1) % len(self.images_paths)
            self.display_current_image()
            self.clear_results()
    
    def clear_images(self):
        self.images_paths = []
        self.images_listbox.delete(0, tk.END)
        self.image_label.config(image='')
        self.clear_results()
        self.update_status("Lista limpiada")
    
    def export_results(self):
        if not self.processed_plates:
            messagebox.showwarning("Advertencia", "No hay resultados para exportar")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
            filetypes=[("Archivos de texto", "*.txt"), ("CSV", "*.csv")])
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("REPORTE DE PLACAS GUATEMALA\n")
                    f.write("="*60 + "\n\n")
                    for item in self.processed_plates:
                        f.write(f"Imagen: {item['image']}\n")
                        f.write(f"Método: {item['method']}\n")
                        f.write(f"OCR Original: {item['plate_original']}\n")
                        f.write(f"Placa Corregida: {item['plate_corrected']}\n")
                        f.write(f"Placa Formateada: {item['plate_formatted']}\n")
                        f.write(f"Válida: {'Sí' if item['is_valid'] else 'No'}\n")
                        if item['vehicle_type']:
                            f.write(f"Tipo: {item['vehicle_type']}\n")
                        f.write(f"Hora: {item['timestamp']}\n")
                        f.write("-"*40 + "\n")
                
                self.update_status(f"Exportado a {file_path}")
                messagebox.showinfo("Éxito", "Resultados exportados correctamente")
            except Exception as e:
                messagebox.showerror("Error", f"Error al exportar: {str(e)}")
    
    def update_status(self, message, color=None):
        self.status_bar.config(text=message)
        if color:
            self.status_bar.config(bg=color)
        else:
            self.status_bar.config(bg='#34495e')
        self.root.update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()
    app = PlateReaderApp(root)
    root.mainloop()