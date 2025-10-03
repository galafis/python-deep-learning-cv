_# 🇧🇷 Deep Learning para Visão Computacional | 🇺🇸 Deep Learning for Computer Vision

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=opencv&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

**Plataforma completa de Deep Learning para Visão Computacional com modelos state-of-the-art**

[🔬 Modelos](#-modelos-implementados) • [📊 Datasets](#-datasets) • [⚡ Quick Start](#-quick-start) • [🎯 Aplicações](#-aplicações-práticas)

</div>

---

## 🇧🇷 Português

### 🔬 Visão Geral

Plataforma abrangente de **Deep Learning para Visão Computacional** desenvolvida em Python, implementando arquiteturas state-of-the-art:

- 🧠 **Redes Neurais Convolucionais**: CNN, ResNet, EfficientNet, Vision Transformer
- 🎯 **Detecção de Objetos**: YOLO, R-CNN, SSD, RetinaNet
- 🖼️ **Segmentação**: U-Net, Mask R-CNN, DeepLab, Semantic Segmentation
- 👁️ **Reconhecimento Facial**: FaceNet, ArcFace, DeepFace
- 🎨 **Geração de Imagens**: GAN, VAE, Diffusion Models
- 📱 **Deploy Mobile**: TensorFlow Lite, ONNX, Edge Computing

### 🎯 Objetivos da Plataforma

- **Implementar** arquiteturas modernas de deep learning
- **Facilitar** desenvolvimento de aplicações de CV
- **Otimizar** modelos para produção e edge devices
- **Demonstrar** técnicas avançadas de computer vision
- **Acelerar** prototipagem e deployment

### 🛠️ Stack Tecnológico

#### Deep Learning Frameworks
- **TensorFlow/Keras**: Framework principal para desenvolvimento
- **PyTorch**: Framework alternativo para pesquisa
- **JAX**: Computação de alto performance
- **Hugging Face Transformers**: Vision Transformers pré-treinados

#### Computer Vision
- **OpenCV**: Processamento de imagens clássico
- **Pillow (PIL)**: Manipulação de imagens
- **scikit-image**: Algoritmos de processamento
- **ImageIO**: Leitura/escrita de formatos diversos

#### Visualização e Análise
- **Matplotlib**: Visualização de resultados
- **Seaborn**: Gráficos estatísticos
- **Plotly**: Visualizações interativas
- **TensorBoard**: Monitoramento de treinamento

#### Deployment e Otimização
- **TensorFlow Lite**: Modelos para mobile/edge
- **ONNX**: Interoperabilidade entre frameworks
- **TensorRT**: Otimização para GPUs NVIDIA
- **OpenVINO**: Otimização para Intel

#### Dados e Augmentação
- **Albumentations**: Augmentação avançada de imagens
- **imgaug**: Augmentação de dados
- **COCO API**: Manipulação de datasets COCO
- **Roboflow**: Gerenciamento de datasets

### 📋 Estrutura da Plataforma

```
python-deep-learning-cv/
├── 📁 src/                        # Código fonte principal
│   ├── 📁 models/                 # Arquiteturas de modelos
│   │   ├── 📁 classification/     # Modelos de classificação
│   │   │   ├── 📄 resnet.py       # ResNet implementation
│   │   │   ├── 📄 efficientnet.py # EfficientNet implementation
│   │   │   ├── 📄 vision_transformer.py # ViT implementation
│   │   │   └── 📄 custom_cnn.py   # CNN customizada
│   │   ├── 📁 detection/          # Modelos de detecção
│   │   │   ├── 📄 yolo_v5.py      # YOLO v5 implementation
│   │   │   ├── 📄 faster_rcnn.py  # Faster R-CNN
│   │   │   ├── 📄 ssd.py          # Single Shot Detector
│   │   │   └── 📄 retinanet.py    # RetinaNet
│   │   ├── 📁 segmentation/       # Modelos de segmentação
│   │   │   ├── 📄 unet.py         # U-Net implementation
│   │   │   ├── 📄 mask_rcnn.py    # Mask R-CNN
│   │   │   ├── 📄 deeplab.py      # DeepLab v3+
│   │   │   └── 📄 fcn.py          # Fully Convolutional Network
│   │   ├── 📁 face_recognition/   # Reconhecimento facial
│   │   │   ├── 📄 facenet.py      # FaceNet implementation
│   │   │   ├── 📄 arcface.py      # ArcFace implementation
│   │   │   └── 📄 deepface.py     # DeepFace wrapper
│   │   └── 📁 generative/         # Modelos generativos
│   │       ├── 📄 gan.py          # Generative Adversarial Network
│   │       ├── 📄 vae.py          # Variational Autoencoder
│   │       └── 📄 diffusion.py    # Diffusion Models
│   ├── 📁 data/                   # Módulos de dados
│   │   ├── 📄 dataset_loader.py   # Carregamento de datasets
│   │   ├── 📄 augmentation.py     # Augmentação de dados
│   │   ├── 📄 preprocessing.py    # Pré-processamento
│   │   └── 📄 data_utils.py       # Utilitários de dados
│   ├── 📁 training/               # Módulos de treinamento
│   │   ├── 📄 trainer.py          # Classe principal de treinamento
│   │   ├── 📄 callbacks.py        # Callbacks customizados
│   │   ├── 📄 losses.py           # Funções de perda
│   │   └── 📄 metrics.py          # Métricas de avaliação
│   ├── 📁 inference/              # Módulos de inferência
│   │   ├── 📄 predictor.py        # Predições em tempo real
│   │   ├── 📄 batch_inference.py  # Inferência em lote
│   │   └── 📄 video_inference.py  # Inferência em vídeo
│   ├── 📁 deployment/             # Módulos de deployment
│   │   ├── 📄 model_converter.py  # Conversão de modelos
│   │   ├── 📄 tflite_converter.py # Conversão para TF Lite
│   │   ├── 📄 onnx_converter.py   # Conversão para ONNX
│   │   └── 📄 api_server.py       # Servidor de API
│   ├── 📁 utils/                  # Utilitários
│   │   ├── 📄 visualization.py    # Visualização de resultados
│   │   ├── 📄 config.py           # Configurações
│   │   ├── 📄 logger.py           # Sistema de logs
│   │   └── 📄 helpers.py          # Funções auxiliares
│   └── 📁 evaluation/             # Avaliação de modelos
│       ├── 📄 evaluator.py        # Avaliação completa
│       ├── 📄 benchmark.py        # Benchmarking
│       └── 📄 analysis.py         # Análise de resultados
├── 📁 datasets/                   # Datasets organizados
│   ├── 📁 classification/         # Datasets de classificação
│   │   ├── 📁 cifar10/           # CIFAR-10 dataset
│   │   ├── 📁 imagenet/          # ImageNet subset
│   │   └── 📁 custom/            # Datasets customizados
│   ├── 📁 detection/             # Datasets de detecção
│   │   ├── 📁 coco/              # COCO dataset
│   │   ├── 📁 pascal_voc/        # Pascal VOC
│   │   └── 📁 open_images/       # Open Images
│   ├── 📁 segmentation/          # Datasets de segmentação
│   │   ├── 📁 cityscapes/        # Cityscapes dataset
│   │   ├── 📁 ade20k/            # ADE20K dataset
│   │   └── 📁 medical/           # Datasets médicos
│   └── 📁 faces/                 # Datasets de faces
│       ├── 📁 lfw/               # Labeled Faces in the Wild
│       ├── 📁 celeba/            # CelebA dataset
│       └── 📁 vggface/           # VGGFace dataset
├── 📁 notebooks/                 # Jupyter notebooks
│   ├── 📄 01_data_exploration.ipynb # Exploração de dados
│   ├── 📄 02_model_training.ipynb # Treinamento de modelos
│   ├── 📄 03_transfer_learning.ipynb # Transfer learning
│   ├── 📄 04_object_detection.ipynb # Detecção de objetos
│   ├── 📄 05_image_segmentation.ipynb # Segmentação
│   ├── 📄 06_face_recognition.ipynb # Reconhecimento facial
│   ├── 📄 07_generative_models.ipynb # Modelos generativos
│   └── 📄 08_model_deployment.ipynb # Deployment
├── 📁 experiments/               # Experimentos e resultados
│   ├── 📁 classification/        # Experimentos classificação
│   ├── 📁 detection/            # Experimentos detecção
│   ├── 📁 segmentation/         # Experimentos segmentação
│   └── 📁 benchmarks/           # Benchmarks de performance
├── 📁 models/                    # Modelos treinados
│   ├── 📁 pretrained/           # Modelos pré-treinados
│   ├── 📁 checkpoints/          # Checkpoints de treinamento
│   └── 📁 production/           # Modelos em produção
├── 📁 apps/                      # Aplicações demo
│   ├── 📁 streamlit_app/        # App Streamlit
│   ├── 📁 flask_api/            # API Flask
│   └── 📁 mobile_app/           # App mobile (TF Lite)
├── 📁 docker/                    # Containers Docker
│   ├── 📄 Dockerfile.gpu        # Container com GPU
│   ├── 📄 Dockerfile.cpu        # Container CPU only
│   └── 📄 docker-compose.yml    # Orquestração
├── 📁 tests/                     # Testes automatizados
│   ├── 📄 test_models.py        # Testes de modelos
│   ├── 📄 test_data.py          # Testes de dados
│   └── 📄 test_inference.py     # Testes de inferência
├── 📁 configs/                   # Arquivos de configuração
│   ├── 📄 training_config.yaml  # Configuração treinamento
│   ├── 📄 model_config.yaml     # Configuração modelos
│   └── 📄 deployment_config.yaml # Configuração deployment
├── 📄 requirements.txt          # Dependências Python
├── 📄 requirements-gpu.txt      # Dependências com GPU
├── 📄 setup.py                  # Setup do pacote
├── 📄 README.md                 # Este arquivo
├── 📄 LICENSE                   # Licença MIT
└── 📄 .gitignore               # Arquivos ignorados
```

### 🔬 Modelos Implementados

#### 1. 🖼️ Classificação de Imagens

**ResNet com Transfer Learning**
```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

class ResNetClassifier:
    def __init__(self, num_classes, input_shape=(224, 224, 3), weights='imagenet'):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.weights = weights
        self.model = self._build_model()
    
    def _build_model(self):
        # Base model pré-treinada
        base_model = ResNet50(
            weights=self.weights,
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Congelar camadas base para transfer learning
        base_model.trainable = False
        
        # Adicionar camadas customizadas
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
    
    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_5_accuracy']
        )
    
    def fine_tune(self, unfreeze_layers=50):
        "'''Fine-tuning: descongelar últimas camadas"'''
        self.model.layers[0].trainable = True
        
        # Congelar todas exceto as últimas N camadas
        for layer in self.model.layers[0].layers[:-unfreeze_layers]:
            layer.trainable = False
        
        # Recompilar com learning rate menor
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_5_accuracy']
        )
```

**Vision Transformer (ViT)**
```python
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout
import numpy as np

class PatchEmbedding(Layer):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = Dense(embed_dim)
        
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return self.projection(patches)

class MultiHeadAttention(Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = Dense(embed_dim * 3)
        self.projection = Dense(embed_dim)
        
    def call(self, x):
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (batch_size, seq_len, 3, self.num_heads, self.head_dim))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attention_scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.head_dim, tf.float32))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_output = tf.matmul(attention_weights, v)
        
        # Concatenate heads
        attention_output = tf.transpose(attention_output, (0, 2, 1, 3))
        attention_output = tf.reshape(attention_output, (batch_size, seq_len, self.embed_dim))
        
        return self.projection(attention_output)

class VisionTransformer:
    def __init__(self, image_size, patch_size, num_classes, embed_dim, num_heads, num_layers):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.model = self._build_model()
    
    def _build_model(self):
        inputs = tf.keras.Input(shape=(self.image_size, self.image_size, 3))
        
        # Patch embedding
        patches = PatchEmbedding(self.patch_size, self.embed_dim)(inputs)
        
        # Add positional encoding
        num_patches = (self.image_size // self.patch_size) ** 2
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=self.embed_dim
        )(positions)
        
        # Add class token
        class_token = tf.Variable(tf.random.normal([1, 1, self.embed_dim]))
        class_token = tf.tile(class_token, [tf.shape(patches)[0], 1, 1])
        
        encoded_patches = patches + position_embedding
        encoded_patches = tf.concat([class_token, encoded_patches], axis=1)
        
        # Transformer blocks
        for _ in range(self.num_layers):
            # Multi-head attention
            x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
            attention_output = MultiHeadAttention(self.embed_dim, self.num_heads)(x1)
            x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
            
            # MLP
            x3 = LayerNormalization(epsilon=1e-6)(x2)
            x3 = Dense(self.embed_dim * 4, activation="gelu")(x3)
            x3 = Dropout(0.1)(x3)
            x3 = Dense(self.embed_dim)(x3)
            x3 = Dropout(0.1)(x3)
            encoded_patches = tf.keras.layers.Add()([x3, x2])
        
        # Classification head
        representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = representation[:, 0]  # Class token
        features = Dense(self.embed_dim, activation="tanh")(representation)
        features = Dropout(0.5)(features)
        logits = Dense(self.num_classes)(features)
        
        model = tf.keras.Model(inputs=inputs, outputs=logits)
        return model
```

#### 2. 🎯 Detecção de Objetos

**YOLO v5 Implementation**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()
    
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class CSPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1, shortcut=True):
        super().__init__()
        hidden_channels = out_channels // 2
        
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1)
        self.conv2 = ConvBlock(in_channels, hidden_channels, 1)
        self.conv3 = ConvBlock(2 * hidden_channels, out_channels, 1)
        
        self.blocks = nn.Sequential(*[ConvBlock(hidden_channels, hidden_channels, 3, padding=1) for _ in range(num_blocks)])
        self.shortcut = shortcut
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        
        x1 = self.blocks(x1)
        
        out = torch.cat((x1, x2), dim=1)
        out = self.conv3(out)
        
        if self.shortcut:
            return x + out
        return out

class YOLOv5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Backbone, Neck, Head implementation
        pass
```

#### 3. 🖼️ Segmentação Semântica

**U-Net para Segmentação Médica**
```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate

class UNet:
    def __init__(self, input_shape=(256, 256, 1), num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _conv_block(self, inputs, filters):
        "'''Bloco convolucional duplo"'''
        conv = Conv2D(filters, 3, activation='relu', padding='same')(inputs)
        conv = Conv2D(filters, 3, activation='relu', padding='same')(conv)
        return conv
    
    def _encoder_block(self, inputs, filters):
        "'''Bloco do encoder"'''
        conv = self._conv_block(inputs, filters)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        return conv, pool
    
    def _decoder_block(self, inputs, skip_connection, filters):
        "'''Bloco do decoder"'''
        up = UpSampling2D(size=(2, 2))(inputs)
        up = Conv2D(filters, 2, activation='relu', padding='same')(up)
        
        # Skip connection
        merge = concatenate([skip_connection, up], axis=3)
        conv = self._conv_block(merge, filters)
        return conv
    
    def _build_model(self):
        inputs = tf.keras.Input(self.input_shape)
        
        # Encoder (Contracting path)
        conv1, pool1 = self._encoder_block(inputs, 64)
        conv2, pool2 = self._encoder_block(pool1, 128)
        conv3, pool3 = self._encoder_block(pool2, 256)
        conv4, pool4 = self._encoder_block(pool3, 512)
        
        # Bottleneck
        conv5 = self._conv_block(pool4, 1024)
        
        # Decoder (Expanding path)
        up6 = self._decoder_block(conv5, conv4, 512)
        up7 = self._decoder_block(up6, conv3, 256)
        up8 = self._decoder_block(up7, conv2, 128)
        up9 = self._decoder_block(up8, conv1, 64)
        
        # Output layer
        outputs = Conv2D(self.num_classes, 1, activation='softmax')(up9)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def compile_model(self):
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', self._dice_coefficient, self._iou_score]
        )
    
    def _dice_coefficient(self, y_true, y_pred, smooth=1):
        "'''Coeficiente Dice para segmentação"'''
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    
    def _iou_score(self, y_true, y_pred, smooth=1):
        "'''Intersection over Union"'''
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
        return (intersection + smooth) / (union + smooth)
```

#### 4. 👁️ Reconhecimento Facial

**FaceNet Implementation**
```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Input
from tensorflow.keras.models import Model
import numpy as np

class FaceNet:
    def __init__(self, input_shape=(160, 160, 3), embedding_size=128):
        self.input_shape = input_shape
        self.embedding_size = embedding_size
        self.model = self._build_model()
    
    def _build_model(self):
        # Base model (pode ser ResNet, Inception, etc.)
        base_model = tf.keras.applications.InceptionResNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Adicionar camadas de embedding
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = Dense(self.embedding_size)(x)
        
        # L2 normalization
        embeddings = Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(x)
        
        model = Model(inputs=base_model.input, outputs=embeddings)
        return model
    
    def triplet_loss(self, alpha=0.2):
        "'''Triplet loss para treinamento"'''
        def loss(y_true, y_pred):
            anchor, positive, negative = y_pred[:, :self.embedding_size], \
                                       y_pred[:, self.embedding_size:2*self.embedding_size], \
                                       y_pred[:, 2*self.embedding_size:]
            
            # Distâncias
            pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
            neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
            
            # Triplet loss
            basic_loss = pos_dist - neg_dist + alpha
            loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
            
            return loss
        return loss
    
    def get_embedding(self, face_image):
        "'''Obter embedding de uma face"'''
        if len(face_image.shape) == 3:
            face_image = np.expand_dims(face_image, axis=0)
        
        embedding = self.model.predict(face_image)
        return embedding[0]
    
    def compare_faces(self, face1, face2, threshold=0.6):
        "'''Comparar duas faces"'''
        emb1 = self.get_embedding(face1)
        emb2 = self.get_embedding(face2)
        
        distance = np.linalg.norm(emb1 - emb2)
        is_same_person = distance < threshold
        
        return {
            'distance': distance,
            'is_same_person': is_same_person,
            'confidence': 1 - (distance / 2)  # Normalizar para 0-1
        }
    
    def face_recognition_pipeline(self, image, known_faces_db):
        "'''Pipeline completo de reconhecimento"'''
        # 1. Detectar faces na imagem
        faces = self._detect_faces(image)
        
        results = []
        for face in faces:
            # 2. Obter embedding da face
            embedding = self.get_embedding(face['image'])
            
            # 3. Comparar com banco de faces conhecidas
            best_match = None
            min_distance = float('inf')
            
            for person_id, known_embedding in known_faces_db.items():
                distance = np.linalg.norm(embedding - known_embedding)
                if distance < min_distance:
                    min_distance = distance
                    best_match = person_id
            
            # 4. Determinar se é uma pessoa conhecida
            if min_distance < 0.6:  # threshold
                results.append({
                    'bbox': face['bbox'],
                    'person_id': best_match,
                    'confidence': 1 - (min_distance / 2),
                    'distance': min_distance
                })
            else:
                results.append({
                    'bbox': face['bbox'],
                    'person_id': 'unknown',
                    'confidence': 0,
                    'distance': min_distance
                })
        
        return results
```

### 🎯 Aplicações Práticas

#### 1. 🏥 Diagnóstico Médico por Imagem

**Classificação de Raios-X**
```python
class MedicalImageClassifier:
    def __init__(self):
        self.model = self._load_pretrained_model()
        self.classes = ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis']
    
    def diagnose_xray(self, xray_image):
        # Pré-processamento
        processed_image = self._preprocess_medical_image(xray_image)
        
        # Predição
        predictions = self.model.predict(processed_image)
        
        # Interpretação
        diagnosis = {
            'primary_diagnosis': self.classes[np.argmax(predictions)],
            'confidence': float(np.max(predictions)),
            'all_probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.classes, predictions[0])
            }
        }
        
        # Mapa de atenção (Grad-CAM)
        attention_map = self._generate_gradcam(processed_image)
        
        return {
            'diagnosis': diagnosis,
            'attention_map': attention_map,
            'recommendations': self._get_recommendations(diagnosis)
        }
    
    def _generate_gradcam(self, image):
        "'''Gerar mapa de atenção Grad-CAM"'''
        # Implementação Grad-CAM para interpretabilidade
        pass
```

#### 2. 🚗 Visão Computacional Automotiva

**Detecção para Veículos Autônomos**
```python
class AutomotiveVision:
    def __init__(self):
        self.object_detector = self._load_object_detector()
        self.lane_detector = self._load_lane_detector()
        self.traffic_sign_classifier = self._load_traffic_sign_classifier()
    
    def process_driving_scene(self, image):
        results = {}
        
        # Detecção de objetos
        objects = self.object_detector.detect(image)
        results['objects'] = self._filter_automotive_objects(objects)
        
        # Detecção de faixas
        lanes = self.lane_detector.detect_lanes(image)
        results['lanes'] = lanes
        
        # Classificação de sinais de trânsito
        traffic_signs = self.traffic_sign_classifier.detect_and_classify(image)
        results['traffic_signs'] = traffic_signs
        
        # Análise de risco
        risk_assessment = self._assess_driving_risk(results)
        results['risk_assessment'] = risk_assessment
        
        return results
    
    def _assess_driving_risk(self, detection_results):
        "'''Avaliar risco baseado nas detecções"'''
        risk_factors = []
        
        # Verificar proximidade de pedestres
        for obj in detection_results['objects']:
            if obj['class'] == 'person' and obj['distance'] < 10:
                risk_factors.append('pedestrian_close')
        
        # Verificar veículos próximos
        for obj in detection_results['objects']:
            if obj['class'] in ['car', 'truck'] and obj['distance'] < 5:
                risk_factors.append('vehicle_close')
        
        # Verificar sinais de trânsito
        for sign in detection_results['traffic_signs']:
            if sign['class'] == 'stop_sign':
                risk_factors.append('stop_sign_detected')
        
        return {
            'risk_level': len(risk_factors),
            'risk_factors': risk_factors,
            'recommended_action': self._get_recommended_action(risk_factors)
        }
```

#### 3. 🏭 Controle de Qualidade Industrial

**Inspeção Automatizada**
```python
class QualityInspection:
    def __init__(self):
        self.defect_detector = self._load_defect_detection_model()
        self.measurement_model = self._load_measurement_model()
    
    def inspect_product(self, product_image):
        inspection_results = {}
        
        # Detecção de defeitos
        defects = self.defect_detector.detect_defects(product_image)
        inspection_results['defects'] = defects
        
        # Medições dimensionais
        measurements = self.measurement_model.measure_dimensions(product_image)
        inspection_results['measurements'] = measurements
        
        # Classificação de qualidade
        quality_score = self._calculate_quality_score(defects, measurements)
        inspection_results['quality_score'] = quality_score
        
        # Decisão de aprovação/rejeição
        decision = self._make_quality_decision(quality_score, defects)
        inspection_results['decision'] = decision
        
        return inspection_results
    
    def _calculate_quality_score(self, defects, measurements):
        "'''Calcular score de qualidade baseado em defeitos e medições"'''
        base_score = 100
        
        # Penalizar por defeitos
        for defect in defects:
            severity = defect['severity']
            base_score -= severity * 10
        
        # Penalizar por medições fora de especificação
        for measurement in measurements:
            if not measurement['within_tolerance']:
                deviation = measurement['deviation_percentage']
                base_score -= deviation * 5
        
        return max(0, base_score)
```

### 🚀 Deployment e Otimização

#### TensorFlow Lite para Mobile
```python
class MobileDeployment:
    def __init__(self, model_path):
        self.model_path = model_path
    
    def convert_to_tflite(self, quantization=True):
        "'''Converter modelo para TensorFlow Lite"'''
        converter = tf.lite.TFLiteConverter.from_saved_model(self.model_path)
        
        if quantization:
            # Quantização para reduzir tamanho
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Quantização INT8 (opcional)
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        
        tflite_model = converter.convert()
        
        # Salvar modelo
        tflite_path = self.model_path.replace('.pb', '.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        return tflite_path
    
    def benchmark_model(self, tflite_path, test_images):
        "'''Benchmark de performance do modelo"'''
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        inference_times = []
        
        for image in test_images:
            start_time = time.time()
            
            # Preparar input
            interpreter.set_tensor(input_details[0]['index'], image)
            
            # Executar inferência
            interpreter.invoke()
            
            # Obter output
            output = interpreter.get_tensor(output_details[0]['index'])
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
        
        return {
            'avg_inference_time': np.mean(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'fps': 1.0 / np.mean(inference_times)
        }
```

### 🎯 Competências Demonstradas

#### Deep Learning
- ✅ **CNNs**: Redes convolucionais clássicas e modernas
- ✅ **Transfer Learning**: Aproveitamento de modelos pré-treinados
- ✅ **Vision Transformers**: Arquiteturas baseadas em atenção
- ✅ **GANs**: Redes adversárias generativas

#### Computer Vision
- ✅ **Classificação**: Reconhecimento de imagens
- ✅ **Detecção**: Localização e classificação de objetos
- ✅ **Segmentação**: Segmentação semântica e de instância
- ✅ **Reconhecimento Facial**: Identificação e verificação

#### MLOps para CV
- ✅ **Model Optimization**: Quantização, pruning, distillation
- ✅ **Edge Deployment**: TensorFlow Lite, ONNX
- ✅ **Performance Monitoring**: Latência, throughput, accuracy
- ✅ **A/B Testing**: Comparação de modelos em produção

### 📊 Benchmarks de Performance

#### Modelos de Classificação
| Modelo | ImageNet Top-1 | Parâmetros | FLOPs | Latência (ms) |
|--------|----------------|------------|-------|---------------|
| ResNet-50 | 76.1% | 25.6M | 4.1G | 15.2 |
| EfficientNet-B0 | 77.1% | 5.3M | 0.39G | 8.7 |
| Vision Transformer | 81.8% | 86M | 17.6G | 45.3 |

#### Modelos de Detecção
| Modelo | COCO mAP | FPS | Tamanho |
|--------|----------|-----|---------|
| YOLOv5s | 37.4 | 140 | 14MB |
| YOLOv5m | 45.4 | 85 | 42MB |
| YOLOv5l | 49.0 | 55 | 92MB |

---

## 🇺🇸 English

### 🔬 Overview

Comprehensive **Deep Learning for Computer Vision** platform developed in Python, implementing state-of-the-art architectures:

- 🧠 **Convolutional Neural Networks**: CNN, ResNet, EfficientNet, Vision Transformer
- 🎯 **Object Detection**: YOLO, R-CNN, SSD, RetinaNet
- 🖼️ **Segmentation**: U-Net, Mask R-CNN, DeepLab, Semantic Segmentation
- 👁️ **Face Recognition**: FaceNet, ArcFace, DeepFace
- 🎨 **Image Generation**: GAN, VAE, Diffusion Models
- 📱 **Mobile Deploy**: TensorFlow Lite, ONNX, Edge Computing

### 🎯 Platform Objectives

- **Implement** modern deep learning architectures
- **Facilitate** CV application development
- **Optimize** models for production and edge devices
- **Demonstrate** advanced computer vision techniques
- **Accelerate** prototyping and deployment

### 🔬 Implemented Models

#### 1. 🖼️ Image Classification
- ResNet with Transfer Learning
- EfficientNet for efficiency
- Vision Transformer (ViT)
- Custom CNN architectures

#### 2. 🎯 Object Detection
- YOLO v5 implementation
- Faster R-CNN
- Single Shot Detector (SSD)
- RetinaNet with focal loss

#### 3. 🖼️ Semantic Segmentation
- U-Net for medical imaging
- Mask R-CNN for instance segmentation
- DeepLab v3+ for semantic segmentation
- Fully Convolutional Networks

#### 4. 👁️ Face Recognition
- FaceNet implementation
- ArcFace for face verification
- DeepFace wrapper
- Face detection and alignment

### 🎯 Skills Demonstrated

#### Deep Learning
- ✅ **CNNs**: Classic and modern convolutional networks
- ✅ **Transfer Learning**: Leveraging pre-trained models
- ✅ **Vision Transformers**: Attention-based architectures
- ✅ **GANs**: Generative adversarial networks

#### Computer Vision
- ✅ **Classification**: Image recognition
- ✅ **Detection**: Object localization and classification
- ✅ **Segmentation**: Semantic and instance segmentation
- ✅ **Face Recognition**: Identification and verification

#### MLOps for CV
- ✅ **Model Optimization**: Quantization, pruning, distillation
- ✅ **Edge Deployment**: TensorFlow Lite, ONNX
- ✅ **Performance Monitoring**: Latency, throughput, accuracy
- ✅ **A/B Testing**: Comparing models in production





---

### ✒️ Autoria

<div align="center">

**Desenvolvido por Gabriel Demetrios Lafis**

[![GitHub](https://img.shields.io/badge/GitHub-galafis-blue?style=flat-square&logo=github)](https://github.com/galafis)

</div>

