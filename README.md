# Sentiment Analysis


This is the last part of three of the Sentiment Analysis Project.
If you haven't seen the other models at the end I will leave the link to the other two models.

In this project, we will delve into three Natural Language Processing (NLP) models, focusing specifically on sentiment analysis. We will explore three distinct architectures: Fully Connected Neural Network (FCNN), Long Short-Term Memory (LSTM), and finally, the Transformers architecture using the BERT model.

The main objectives of this project are as follows:

- Evaluate the performance of the three models in relation to the sentiment analysis task.
- Assess the level of complexity involved in building each architecture.
- Compare the training times required for each model. 

Upon completion of this study, we aim to draw informed conclusions about which model proves most suitable for sentiment analysis. To achieve this, we will employ a dataset with six classes for sentiment classification.<br>
In all three projects we will use the same database but manipulate it with different techniques.


# Model 3: Transformers (BERT) 


The Transformer architecture, introduced in the paper "Attention is All You Need" in 2017 (<a href="https://arxiv.org/abs/1706.03762">Attention is All You Need</a>), is a significant innovation in the field of natural language processing (NLP). What makes it revolutionary compared to previous approaches are several fundamental features. Firstly, Transformer introduced the concept of attention mechanism. This mechanism allows the model to assign different weights to different parts of the input, focusing on the relevant parts during output generation. This solves the gradient fading problem found in models such as recurrent neural networks (RNNs) and convolutional neural networks (CNNs), allowing the model to capture long-range relationships in sequences.<br>
Furthermore, the completely attention-based structure of Transformers allows for more efficient parallelization compared to RNNs and CNNs. This means Transformers can be trained faster and scaled to larger data sets, making them more practical for real-world applications. Another crucial aspect is the use of pre-trained language models, such as GPT (Generative Pre-trained Transformer) and BERT (Bidirectional Encoder Representations from Transformers), in this last part of the project we will use BERT. These pre-trained models can be fine-tuned for specific tasks with relatively little training data, making them highly efficient in a variety of NLP tasks such as text classification, machine translation, and text generation. Transformers have the ability to model long-range contexts in text sequences. This means they can capture complex semantic relationships and temporal dependencies, making them suitable for a wide range of NLP tasks that require a deep understanding of context.<br>
I made a small introduction briefly explaining how the attention function works without using a framework. If you want to follow along, just click on the link <a href="https://github.com/CoyoteColt/Attention-Is-All-You-Need">Attention is All You Need-A Brief explanation</a>

<img src="https://maquinasqueaprendem.files.wordpress.com/2023/05/transformers.jpg" alt="Attention Mechanism">

## Techniques used for data processing in the Transformers model

- ## Spacy
In all models we will use Spacy as a way to simplify the text by removing very repetitive words.
- ## Custom tokenization
Subword tokenization; This technique breaks down words into smaller subunits, such as prefixes, suffixes, or meaningful parts of the word. This is particularly useful when dealing with rare or unknown words, as it allows you to represent them in terms of known units. By capturing the morphological structure of words, this approach helps the model better understand the text.<br>
Piece-wise tokenization; In this technique, text is divided into units called pieces, which can be complete words, subwords or even individual characters. An algorithm such as Byte Pair Encoding (BPE) or SentencePiece is often used to perform this division. This approach is crucial for dealing with languages ​​that have extensive vocabularies and a wide variety of morphological forms, as it allows for a more flexible representation of the text.

- ## Huggingface platform
The Hugging Face platform is an open source library that offers a wide range of pre-trained language models and tools for working with them. It significantly simplifies the process of importing and using these models in natural language processing (NLP) projects. With Hugging Face, developers can easily import pre-trained models using just a few lines of code, enabling rapid prototyping and experimentation on a variety of NLP tasks.

- ## Token CLS
The CLS token in BERT (Bidirectional Encoder Representations from Transformers) is a special tag added to the input during preprocessing. It denotes the beginning of a sequence and is mainly used for text classification tasks. During BERT training, the final token representation CLS is used as a contextual representation of the entire sequence, and it is this representation that is fed to downstream classification tasks such as text classification or sentiment analysis. BERT learns to encode relevant information about the entire sequence into this special representation CLS, making it useful as a kind of contextual summary of the input

- ## Stratified sample
Stratified sampling is a technique used in statistics to ensure that subgroups (or strata) of a population are adequately represented within a sample. It is particularly useful in situations where the population is heterogeneous and subgroups have different characteristics that are important for research.

- ## TFDistilBertModel
It is a class within the transformers library. It represents the DistilBERT model adapted for TensorFlow. DistilBERT is a simplified version of BERT that offers a good balance between performance and efficiency. It has been trained to distill knowledge from BERT while retaining most of its effectiveness, but with a lighter, faster architecture for training and inference.

- ## trainable = False
This statement sets the trainable property of the specified layer to False. When a layer is defined as untrainable, this means that its weights will not be updated during model training. In other words, the state of this layer will remain the same, regardless of the learning process that the rest of the model is going through.

- ## Callbacks end Early Stopping
Another technique that will also be used in the three models, Callbacks end Early Stopping are functions that allow you to monitor and control the training process of machine learning models. They provide a communication channel between the model and the user code, allowing the execution of personalized actions at specific points in the training, such as adjusting the learning rate function and stopping training after the model reaches its plateau.

## Results

<img src="https://cdn.discordapp.com/attachments/809675955689881640/1229102166908735619/image.png?ex=662e7598&is=661c0098&hm=0600c288ab5b2bd22b7367bdca376838ae946605036fafdc61cd15a128b4f2f4&" alt="Model Transformers"><br>
<img src="https://cdn.discordapp.com/attachments/809675955689881640/1229101824796266596/image.png?ex=662e7546&is=661c0046&hm=9d2bef230bbae92e2a426cd5c7a7a2edc765190e5eda8e4fedcc8e76582c49bf&" alt="Model Transformers"><br>
<img src="https://cdn.discordapp.com/attachments/809675955689881640/1229102483419168799/image.png?ex=662e75e3&is=661c00e3&hm=642879e9ae7d5909030a655f5dad04669007c7024868b20a6bcad371697be3c4&" alt="Model Transformers"><br>

- In our last model we achieved an accuracy of 92% with almost 40 minutes of CPU training, without a doubt the best model, right? Or maybe not?.

In this sentiment analysis project we approached three different architectures, a simpler one which was FCNN, an intermediate one which we used LSTM and finally we used the transformer architecture with the BERT model, clearly the transformers architecture was the best, obtaining 92% accuracy. But is using a transformer really a better alternative? in the first model we had a very good result and it took only 2 minutes for training and an accuracy of 83% compared to the BERT model, 40 minutes were spent using CPU (8 minutes with GPU), it is a much higher computational cost and, depending your needs, and it is better to use a simpler model that will cost much less to train, will be ready much faster and will still perform relatively well. For work environments that require very high precision, such as areas of medicine, then it is recommended to go for a model with higher precision, basically everything will depend on your needs.

<br>

## If you want to reproduce the experiment on your machine, below are the versions used
It is worth noting that I trained this last model in a Linux environment since the version of Tensorflow 2.15.0 at the time I created the project did not have support for GPU for Windows. But for information purposes, I ran the model on the CPU to get an idea of ​​the total time.

<img src="https://cdn.discordapp.com/attachments/809675955689881640/1227734807627173938/image.png?ex=66297c24&is=66170724&hm=1762e9a8a1238c8f981d2e78ddd6ab538bf4f12b31d63a1ecd702ea49eed83f4&" alt="version"><br>





- Link to the first two models

<a href="https://github.com/CoyoteColt/Sentiment-Analysis-FCNN">Model 1: Fully Connected Neural Network (FCNN)</a>

<a href="https://github.com/CoyoteColt/Sentiment-Analysis-LSTM">Model 2: Long Short-Term Memory (LSTM)</a>
