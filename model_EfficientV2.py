import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from keras.layers import GlobalAveragePooling2D
import time
import numpy as np
from keras.callbacks import EarlyStopping
import multiprocessing
from keras.applications.efficientnet_v2 import EfficientNetV2


workers = multiprocessing.cpu_count()



sdir=r'archive\Image Data base\Image Data base'
min_samples= 300 # set limit for minimum images a class must have to be included in the dataframe
filepaths = []
labels=[]

def filter_data(sdir, min_samples, filepaths, labels):
    classlist=os.listdir(sdir)   
    for klass in classlist:
        classpath=os.path.join(sdir, klass)
        flist=os.listdir(classpath)
        if len(flist) >= min_samples:
            for f in flist:
                fpath=os.path.join(classpath,f)
                filepaths.append(fpath)
                labels.append(klass)
        else:
            print('class ', klass, ' has only', len(flist), ' samples and will not be included in dataframe')

filter_data(sdir, min_samples, filepaths, labels)

def holdout(filepaths, labels):
    Fseries=pd.Series(filepaths, name='filepaths')
    Lseries=pd.Series(labels, name='labels')        
    df=pd.concat([Fseries, Lseries], axis=1)


    train_df, dummy_df=train_test_split(df, train_size=.7, shuffle=True, random_state=123, stratify=df['labels'])

    valid_df, test_df=train_test_split(dummy_df, train_size=.33, shuffle=True, random_state=123, stratify=dummy_df['labels'])

    print('train_df lenght: ', len(train_df), '  test_df length: ', len(test_df), '  valid_df length: ', len(valid_df))
    return (train_df, valid_df, test_df)

train_df, valid_df, test_df = holdout(filepaths, labels)

# get the number of classes and the images count for each class in train_df
classes=sorted(list(train_df['labels'].unique()))
class_count = len(classes)
print('The number of classes in the dataset is: ', class_count)
groups=train_df.groupby('labels')
print('{0:^30s} {1:^13s}'.format('CLASS', 'IMAGE COUNT'))
countlist=[]
classlist=[]

for label in sorted(list(train_df['labels'].unique())):
    group=groups.get_group(label)
    countlist.append(len(group))
    classlist.append(label)
    print('{0:^30s} {1:^13s}'.format(label, str(len(group))))

# get the classes with the minimum and maximum number of train images
max_value=np.max(countlist)
max_index=countlist.index(max_value)
max_class=classlist[max_index]
min_value=np.min(countlist)
min_index=countlist.index(min_value)
min_class=classlist[min_index]

print(max_class, ' has the most images= ',max_value, ' ', min_class, ' has the least images= ', min_value)

# lets get the average height and width of a sample of the train images
ht=0
wt=0

# select 100 random samples of train_df
train_df_sample=train_df.sample(n=100, random_state=123,axis=0)

for i in range (len(train_df_sample)):
    fpath=train_df_sample['filepaths'].iloc[i]
    img=plt.imread(fpath)
    shape=img.shape
    ht += shape[0]
    wt += shape[1]

print('average height= ', ht//100, ' average width= ', wt//100, 'aspect ratio= ', ht/wt)

# Definindo tamanho do batch e tamanho da imagem
batch_size = 32
img_size = (224, 224)

# Criando geradores de imagens para treinamento e validação
train_gen = ImageDataGenerator(
    horizontal_flip=True, 
    rotation_range=20, 
    width_shift_range=.2,
    height_shift_range=.2, 
    zoom_range=.2
).flow_from_dataframe(
    train_df, 
    x_col='filepaths', 
    y_col='labels', 
    target_size=img_size,
    class_mode='categorical', 
    color_mode='rgb', 
    shuffle=True, 
    batch_size=batch_size
)

valid_gen = ImageDataGenerator().flow_from_dataframe(
    valid_df, 
    x_col='filepaths', 
    y_col='labels', 
    target_size=img_size,
    class_mode='categorical', 
    color_mode='rgb', 
    shuffle=False, 
    batch_size=batch_size
)

# Calculando tamanho do batch para conjunto de teste
length = len(test_df)
test_batch_size = 32
test_steps = int(length / test_batch_size)

# Criando gerador de imagens para teste
test_gen = ImageDataGenerator().flow_from_dataframe(
    test_df, 
    x_col='filepaths', 
    y_col='labels', 
    target_size=img_size,
    class_mode='categorical', 
    color_mode='rgb', 
    shuffle=False, 
    batch_size=test_batch_size
)

# Obtendo informações relevantes dos geradores
classes = list(train_gen.class_indices.keys())
class_indices = list(train_gen.class_indices.values())
class_count = len(classes)
labels = test_gen.labels


# Imprimindo informações úteis
print('test batch size: ', test_batch_size, '  test steps: ', test_steps, ' number of classes : ', class_count)

img_shape=(img_size[0], img_size[1], 3)

# Define o modelo base

base_model = tf.keras.applications.EfficientNetV2(
    weights='imagenet', 
    include_top=False,
    input_shape=img_shape,
    pooling='max')

# Adiciona camadas adicionais para adaptação ao conjunto de dados
x = base_model.output
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(rate=.4, seed=123)(x)


predictions = Dense(class_count, activation='softmax')(x)

# Define o modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# Congela as camadas do modelo base para que não sejam treinadas durante o treinamento
for layer in base_model.layers:
    layer.trainable = False

# Compila o modelo com uma função de perda e um otimizador
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#define o earlyStopping
earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# Treina o modelo
model.fit(train_gen, validation_data=valid_gen, epochs=2, callbacks=[earlystop], workers = workers)

# Avalia o modelo no conjunto de teste
model.evaluate(test_gen)