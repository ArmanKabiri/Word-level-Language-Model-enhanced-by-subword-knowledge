# Word-level RNN-based Language_Model boosted with character-level features

### This language model is written in PyTorch.

#### To train it on your own corpus, simply call Train.py with its arguments.

The arguments are as follows:

* '--corpus_train_file' : 'location of the data corpus for training'
* '--corpus_valid_file': 'location of the data corpus for validation'
* '--embeddings_file' : 'If pretrained embeddings exist, load them here.'
* '--output_model_path' : 'Path to save the trained model.'
* '--output_id2word_path' : 'Path to save dictionary file (id2word)'
* '--output_word2id_path' : 'Path to save dictionary file (word2id)'
* '--n_layers' : 'Number of LSTM layers stacked on top of each other.'
* '--hidden_size' : 'Number of hidden units in each LSTM layer'
* '--dropout_probablity' : 'Dropout probablity applied on embeddings layer and LSTM layer.'
* '--embeddings_dim' : 'The dimension of the embeddings'
* '--batch_size' : 'Number of samples per batch.'
* '--seq_len' : 'Length of the sequence for back propagation.'
* '--epochs' : 'Number of epocks.'
* '--lr' : 'Learning rate.'
* '--seed' : 'The seed for randomness'
* '--clip_grad' : 'Clip gradients during training to prevent exploding gradients.'
* '--print_steps' : 'Print training info every n steps.'
* '--bidirectional_model' : 'Use it if you want your LSTM to be bidirectional.'
* '--tie_weights' : 'Tie weights of the last decoder layer to the embeddings layer.'
* '--freez_embeddings' : 'Prevent the pretrained loaded embeddings from fine-tuning.'
* '--gpu' : 'Turn it on if you have a GPU device.'


#### To test the model with generating a new text, simply call Test.py with the following arguments:

* '--model_path' : 'Path to the pre-trained model.'
* '--id2word_path' : 'Path to the dictionary file (id2word)'
* '--word2id_path' : 'Path to the dictionary file (word2id)'
* '--seed_word' : 'The seed word to generate a new text'
* '--gpu' : 'Turn it one if you have a GPU device'


#### Feel free to contribute to this model.
